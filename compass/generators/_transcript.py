"""Dream transcripts -- cross-session learning for any generator.

A transcript captures two things:
    1. What the human said (instructions/questions)
    2. What the system did (actions/steps/facts)

When a similar instruction arrives later, the transcript is loaded
as DomainSection context. The model reads its past experience and
knows what to do.

This module is generator-agnostic. Each project (neo, trinity, etc.)
provides its own transcripts_dir and optionally overrides how entries
are flattened to searchable text.

Usage:
    store = DreamStore(Path("./transcripts"))
    store.save("linkedin_messaging", entries, target="Safari")
    store.build_index()

    matches = store.search("message a contact on linkedin")
    for name, score, transcript in matches:
        section = store.to_context(transcript)
        # -> DomainSection for injection into GenerationContext
"""

from __future__ import annotations

import hashlib
import json
import os
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from compass.generators._types import DomainSection


# ---------------------------------------------------------------------------
# Embedding backend (self-contained, reads same env vars as rag.py)
# ---------------------------------------------------------------------------

_COMPASS_ENV = Path.home() / ".compass" / ".env"


def _load_env_fallback():
    keys = ("LLAMACPP_SERVERS", "OLLAMA_SERVERS", "OLLAMA_HOST", "EMBEDDING_MODEL")
    if any(os.getenv(k) for k in keys):
        return
    if not _COMPASS_ENV.exists():
        return
    for line in _COMPASS_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"')
            if key in keys and not os.getenv(key):
                os.environ[key] = value


_load_env_fallback()

_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b").split("@")[0]


def _get_llamacpp_host() -> Optional[str]:
    servers = os.getenv("LLAMACPP_SERVERS", "")
    for part in servers.split(","):
        part = part.strip()
        if "=" in part:
            name, url = part.split("=", 1)
            if name.strip() == "embed":
                return url.strip()
    return None


def _get_ollama_host() -> str:
    servers = os.getenv("OLLAMA_SERVERS", "")
    if servers:
        for server in servers.split(","):
            server = server.strip()
            if "=" in server:
                name, url = server.split("=", 1)
                if name.strip() in ("local", "big"):
                    return url.strip()
    return os.getenv("OLLAMA_HOST", "http://localhost:11434")


def _embed_texts(texts: List[str]) -> np.ndarray:
    import urllib.request

    host = _get_llamacpp_host()
    if host:
        url = f"{host}/v1/embeddings"
        data = json.dumps({"input": texts}).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            sorted_data = sorted(result["data"], key=lambda d: d["index"])
            return np.array([d["embedding"] for d in sorted_data])

    host = _get_ollama_host()
    url = f"{host}/api/embed"
    data = json.dumps({"model": _EMBED_MODEL, "input": texts}).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
        return np.array(result["embeddings"])


def _embed_query(query: str) -> np.ndarray:
    vecs = _embed_texts([query])
    vec = vecs[0]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ---------------------------------------------------------------------------
# DreamStore -- per-project dream storage and retrieval
# ---------------------------------------------------------------------------


class DreamStore:
    """Dream storage with cached embedding index.

    Each project instantiates its own DreamStore pointing at a transcripts
    directory. The embedding cache lives in a sibling .index/ directory.
    """

    def __init__(self, transcripts_dir: Path, index_dir: Path | None = None):
        self.transcripts_dir = transcripts_dir
        self.index_dir = index_dir or transcripts_dir.parent / ".index"
        self._names: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._hash: str = ""

    # --- CRUD ---

    def save(self, name: str, entries: list, **meta) -> Path:
        """Save a dream transcript.

        entries: list of dicts, each is either:
            {"said": "open linkedin and message a contact"}
            {"did": [list of step/action dicts]}
        meta: optional keys like target="Safari", domain="coding"
        """
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        transcript = {
            "name": name,
            "date": datetime.now().isoformat(),
        }
        transcript.update(meta)
        transcript["entries"] = entries

        path = self.transcripts_dir / f"{name}.yaml"
        with open(path, "w") as f:
            yaml.dump(transcript, f, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)
        print(f"  dream saved: {path}")
        return path

    def load(self, name: str) -> Optional[dict]:
        path = self.transcripts_dir / f"{name}.yaml"
        if not path.exists():
            return None
        with open(path) as f:
            return yaml.safe_load(f)

    def list_names(self) -> List[str]:
        if not self.transcripts_dir.exists():
            return []
        return [p.stem for p in sorted(self.transcripts_dir.glob("*.yaml"))]

    # --- Search ---

    def search(self, query: str, threshold: float = 0.2) -> List[Tuple[str, float, dict]]:
        """Find dreams relevant to a query.

        Returns [(name, score, transcript)] sorted by relevance.
        Uses cached embeddings, falls back to keyword overlap.
        """
        names = self.list_names()
        if not names:
            return []

        transcripts = {}
        texts = {}
        for name in names:
            t = self.load(name)
            if t:
                transcripts[name] = t
                texts[name] = self._flatten(t)

        if not transcripts:
            return []

        # Semantic search via cached index
        try:
            self.build_index()

            if self._embeddings is not None and len(self._names) > 0:
                q_vec = _embed_query(query)
                scores = np.dot(self._embeddings, q_vec)

                results = []
                for i, name in enumerate(self._names):
                    if name in transcripts:
                        score = float(scores[i])
                        if score > threshold:
                            results.append((name, score, transcripts[name]))
                results.sort(key=lambda x: x[1], reverse=True)
                return results

        except Exception:
            pass

        # Fallback: keyword overlap
        query_words = set(query.lower().split())
        results = []
        for name, t in transcripts.items():
            all_words = set(texts[name].lower().split())
            overlap = len(query_words & all_words)
            if overlap > 0:
                results.append((name, overlap / len(query_words), t))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # --- Index ---

    def build_index(self, force: bool = False) -> int:
        """Build or refresh the embedding index. Returns count."""
        current_hash = self._content_hash()

        if not force and self._hash == current_hash and self._embeddings is not None:
            return len(self._names)

        names = self.list_names()
        if not names:
            self._names = []
            self._embeddings = None
            self._hash = current_hash
            return 0

        texts = {}
        for name in names:
            t = self.load(name)
            if t:
                texts[name] = self._flatten(t)

        ordered = list(texts.keys())
        if not ordered:
            self._names = []
            self._embeddings = None
            self._hash = current_hash
            return 0

        # Try disk cache
        if not force and self._load_cache(current_hash, ordered):
            self._hash = current_hash
            return len(self._names)

        # Embed fresh
        embeddings = _embed_texts([texts[n] for n in ordered])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        self._names = ordered
        self._embeddings = embeddings
        self._hash = current_hash
        self._save_cache()
        return len(self._names)

    # --- Context injection ---

    def to_context(self, transcript: dict) -> DomainSection:
        """Convert a dream to a DomainSection for the model."""
        name = transcript.get("name", "unknown")
        lines = [f"# Past experience: {name}"]

        for key in ("target", "domain", "workspace"):
            if transcript.get(key):
                lines.append(f"{key.title()}: {transcript[key]}")
        lines.append("")

        for entry in transcript.get("entries", []):
            if "said" in entry:
                lines.append(f"Human: {entry['said']}")
            if "did" in entry:
                for step in entry["did"]:
                    lines.append(self._format_step(step))
            lines.append("")

        return DomainSection(
            heading=f"Past Experience: {name}",
            content="\n".join(lines),
        )

    def search_and_inject(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.3,
    ) -> List[DomainSection]:
        """Search dreams and return DomainSections ready for context injection."""
        matches = self.search(query, threshold=0.2)
        sections = []
        for name, score, t in matches[:top_k]:
            if score >= threshold:
                sections.append(self.to_context(t))
                print(f"  dream loaded: {name} ({score:.0%} match)")
        return sections

    # --- Overridable ---

    def _flatten(self, transcript: dict) -> str:
        """Flatten a transcript into searchable text."""
        parts = [transcript.get("name", "")]
        for key in ("target", "domain", "workspace"):
            if transcript.get(key):
                parts.append(transcript[key])
        for entry in transcript.get("entries", []):
            if "said" in entry:
                parts.append(entry["said"])
            for step in entry.get("did", []):
                parts.extend(str(v) for v in step.values())
        return " ".join(parts)

    def _format_step(self, step: dict) -> str:
        """Format a single step dict for context display."""
        # Neo-style: action + target/value/url
        action = step.get("action")
        if action:
            detail = step.get("target", step.get("value",
                     step.get("url", "")))
            return f"  {action}: {detail}"

        # Trinity-style: step_id + description + value
        step_id = step.get("step_id", step.get("step", ""))
        desc = step.get("description", "")
        fact_name = step.get("fact", step.get("produces", ""))
        fact_val = step.get("value", step.get("result", ""))

        parts = []
        if step_id:
            parts.append(step_id)
        if desc:
            parts.append(desc)
        if fact_name and fact_val:
            parts.append(f"-> {fact_name}: {fact_val}")
        elif fact_name:
            parts.append(f"-> {fact_name}")
        elif fact_val:
            parts.append(f"-> {fact_val}")

        return f"  {' | '.join(parts)}" if parts else f"  {step}"

    # --- Cache internals ---

    def _content_hash(self) -> str:
        h = hashlib.sha256()
        if self.transcripts_dir.exists():
            for p in sorted(self.transcripts_dir.glob("*.yaml")):
                h.update(p.read_bytes())
        return h.hexdigest()[:16]

    def _cache_path(self) -> Path:
        return self.index_dir / "dream_embeddings.npz"

    def _meta_path(self) -> Path:
        return self.index_dir / "dream_meta.json"

    def _save_cache(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._cache_path(),
                            embeddings=self._embeddings,
                            names=np.array(self._names, dtype=object))
        meta = {"hash": self._hash, "names": self._names}
        with open(self._meta_path(), "w") as f:
            json.dump(meta, f, indent=2)

    def _load_cache(self, expected_hash: str, expected_names: list) -> bool:
        if not self._cache_path().exists() or not self._meta_path().exists():
            return False
        try:
            with open(self._meta_path()) as f:
                meta = json.load(f)
            if meta.get("hash") != expected_hash:
                return False
            data = np.load(self._cache_path(), allow_pickle=True)
            cached = list(data["names"])
            if cached != expected_names:
                return False
            self._names = cached
            self._embeddings = data["embeddings"]
            return True
        except Exception:
            return False
