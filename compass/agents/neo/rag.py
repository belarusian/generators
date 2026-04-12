"""
RAG (Retrieval Augmented Generation) for codebase context.

Embeds code chunks and retrieves relevant context based on query similarity.
"""

import hashlib
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import threading

# Debug mode
DEBUG = (os.getenv("COMPASS_DEBUG", "").lower() in ("1", "true", "yes") or
         os.getenv("DEBUG", "").lower() in ("1", "true", "yes"))

# Embedding backend state
_backend = None  # 'ollama' or 'sentence-transformers'
_st_model = None  # sentence-transformers model (lazy loaded)

# Ollama embedding model -- reads from EMBEDDING_MODEL env (strip @server suffix)
_raw_embed = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b")
OLLAMA_EMBED_MODEL = _raw_embed.split("@")[0]

# Singleton embedder cache -- one CodeEmbedder per resolved project path
_embedders: Dict[str, "CodeEmbedder"] = {}
_embedder_lock = threading.Lock()

# Query embedding cache -- FIFO eviction at 64 entries
_query_cache: Dict[str, np.ndarray] = {}
_QUERY_CACHE_MAX = 64


def _get_llamacpp_embed_host() -> Optional[str]:
    """Get llama.cpp embed server from LLAMACPP_SERVERS (embed=url)."""
    servers = os.getenv("LLAMACPP_SERVERS", "")
    if not servers:
        return None
    for part in servers.split(","):
        part = part.strip()
        if "=" in part:
            name, url = part.split("=", 1)
            if name.strip() == "embed":
                return url.strip()
    return None


def _get_embed_host() -> str:
    """Get Ollama host for embeddings, preferring 'local' to free big for inference."""
    servers = os.getenv("OLLAMA_SERVERS", "")
    if servers:
        parsed = {}
        for server in servers.split(","):
            server = server.strip()
            if "=" in server:
                name, url = server.split("=", 1)
                parsed[name] = url

        # Prefer local -- embeddings are cheap, keep big free for the Oracle
        for prefer in ("local", "big"):
            if prefer in parsed:
                return parsed[prefer]

        # Fall through to first server
        if parsed:
            return next(iter(parsed.values()))

    # Fallback to legacy OLLAMA_HOST
    return os.getenv("OLLAMA_HOST", "http://localhost:11434")


def _test_llamacpp() -> bool:
    """Test if llama.cpp embedding server is reachable (OpenAI-compatible /v1/embeddings)."""
    import urllib.request

    host = _get_llamacpp_embed_host()
    if not host:
        return False

    url = f"{host}/v1/embeddings"
    data = json.dumps({"input": "test"}).encode()

    try:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
            return "data" in result and len(result["data"]) > 0
    except Exception:
        return False


def _test_ollama() -> bool:
    """Test if Ollama is available with the embedding model."""
    import urllib.request
    import urllib.error

    host = _get_embed_host()
    url = f"{host}/api/embeddings"
    data = json.dumps({"model": OLLAMA_EMBED_MODEL, "prompt": "test"}).encode()

    try:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
            return "embedding" in result
    except Exception:
        return False


def _ollama_embed(text: str) -> List[float]:
    """Get embedding from Ollama API."""
    import urllib.request

    host = _get_embed_host()
    url = f"{host}/api/embeddings"
    data = json.dumps({"model": OLLAMA_EMBED_MODEL, "prompt": text}).encode()

    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        return result["embedding"]


def _ollama_embed_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts via Ollama /api/embed (batch endpoint)."""
    import urllib.request

    host = _get_embed_host()
    url = f"{host}/api/embed"
    data = json.dumps({"model": OLLAMA_EMBED_MODEL, "input": texts}).encode()

    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
        return result["embeddings"]


def _llamacpp_embed_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings via llama.cpp /v1/embeddings (OpenAI-compatible)."""
    import urllib.request

    host = _get_llamacpp_embed_host()
    url = f"{host}/v1/embeddings"
    data = json.dumps({"input": texts}).encode()

    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
        # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
        sorted_data = sorted(result["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in sorted_data]


def _init_backend() -> bool:
    """Initialize the embedding backend. Returns False if no backend is available."""
    global _backend, _st_model

    if _backend is not None:
        return True

    # Try llama.cpp first (LLAMACPP_SERVERS embed=...)
    if _test_llamacpp():
        _backend = "llamacpp"
        return True

    # Try Ollama (OLLAMA_SERVERS)
    if _test_ollama():
        _backend = "ollama"
        return True

    return False


def get_embedder(project_path: str) -> "CodeEmbedder":
    """Get or create a singleton CodeEmbedder for the given project path.

    Resolves the path, creates the embedder once, loads from disk on first access.
    Thread-safe via module lock.
    """
    resolved = str(Path(project_path).resolve())
    with _embedder_lock:
        if resolved not in _embedders:
            embedder = CodeEmbedder(resolved)
            embedder._load_embeddings()
            _embedders[resolved] = embedder
        return _embedders[resolved]


def _embed_texts(texts: list) -> Optional[np.ndarray]:
    """Embed a list of texts using the available backend.

    Returns None if no backend is available.
    """
    if not _init_backend():
        return None

    if _backend == "llamacpp":
        return _embed_texts_llamacpp(texts)
    elif _backend == "ollama":
        return _embed_texts_ollama(texts)
    else:
        return _st_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


def _embed_texts_ollama(texts: list) -> np.ndarray:
    """Ollama embedding with batch endpoint + parallel fallback."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    BATCH_SIZE = 32

    # Try batch endpoint first
    try:
        if len(texts) <= BATCH_SIZE:
            return np.array(_ollama_embed_batch(texts))

        # Chunk into batches, run in parallel
        batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
        results = [None] * len(batches)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_ollama_embed_batch, batch): idx for idx, batch in enumerate(batches)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return np.array([emb for batch in results for emb in batch])

    except Exception:
        # Batch endpoint unavailable -- parallel individual calls
        results = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_ollama_embed, text): idx for idx, text in enumerate(texts)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return np.array(results)


def _embed_texts_llamacpp(texts: list) -> np.ndarray:
    """llama.cpp embedding with parallel batches via /v1/embeddings."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    BATCH_SIZE = 32

    if len(texts) <= BATCH_SIZE:
        return np.array(_llamacpp_embed_batch(texts))

    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    results = [None] * len(batches)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_llamacpp_embed_batch, batch): idx for idx, batch in enumerate(batches)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return np.array([emb for batch in results for emb in batch])


def _embed_query(query: str) -> Optional[np.ndarray]:
    """Embed a single query. Cached with FIFO eviction. Returns None if no backend."""
    global _query_cache

    if query in _query_cache:
        return _query_cache[query]

    if not _init_backend():
        return None

    if _backend == "ollama":
        embedding = np.array(_ollama_embed(query))
        norm = np.linalg.norm(embedding)
        result = embedding / norm if norm > 0 else embedding
    else:
        result = _st_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    # FIFO eviction
    if len(_query_cache) >= _QUERY_CACHE_MAX:
        oldest = next(iter(_query_cache))
        del _query_cache[oldest]
    _query_cache[query] = result

    return result


@dataclass
class CodeChunk:
    """A chunk of code with metadata."""
    id: str              # Unique identifier (file:name)
    type: str            # 'function', 'class', 'method'
    name: str            # Function/class name
    file: str            # Relative file path
    line: int            # Starting line number
    content: str         # The actual code/docstring
    signature: str       # Function signature or class definition line


# Gap detection threshold - if max RAG score is below this, consider it a capability gap
GAP_THRESHOLD = 0.3


@dataclass
class RAGResult:
    """Result from RAG retrieval with score metadata for gap detection."""
    context: str                          # Formatted context string
    max_score: float                      # Highest relevance score
    avg_score: float                      # Average of top-k scores
    has_gap: bool                         # True if max_score < GAP_THRESHOLD
    top_matches: List[Tuple[str, float]]  # [(chunk_id, score), ...]


class CodeEmbedder:
    """Embeds code chunks from a codebase."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.compass_dir = self.project_path / ".compass"
        self.embeddings_path = self.compass_dir / "embeddings.npz"
        self.metadata_path = self.compass_dir / "embeddings_meta.json"

        # In-memory state
        self.chunks: Dict[str, CodeChunk] = {}  # id -> chunk
        self.embeddings: Optional[np.ndarray] = None  # shape: (n_chunks, dim)
        self.chunk_ids: List[str] = []  # Maps embedding index to chunk id
        self.missing_file_counter: int = 0  # Counter for missing files across builds
    # Note: This counter tracks how many files have disappeared since the last index build.
    # When it exceeds the threshold, a full rebuild is triggered automatically.

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for change detection."""
        try:
            content = file_path.read_bytes()
            return hashlib.md5(content).hexdigest()[:12]
        except Exception:
            return ""

    def _extract_chunks_from_file(self, file_path: Path) -> List[CodeChunk]:
        """Extract code chunks from a source file."""
        if file_path.suffix == ".py":
            return self._extract_chunks_python(file_path)
        return self._extract_chunks_treesitter(file_path)

    def _extract_chunks_treesitter(self, file_path: Path) -> List[CodeChunk]:
        """Extract code chunks from a non-Python file via tree-sitter."""
        try:
            from compass.agents.neo.treesitter import extract_definitions
        except ImportError:
            return []

        rel_path = str(file_path.relative_to(self.project_path))
        ext = file_path.suffix

        try:
            content = file_path.read_text()
            source = content.encode("utf-8")
        except Exception:
            return []

        defs = extract_definitions(source, ext, rel_path)
        if not defs:
            return []

        lines = content.splitlines()
        result = {}  # (file, line) -> chunk -- dedup

        for d in defs:
            key = (d.file, d.line)
            if key in result and result[key].type == "method":
                continue  # keep more specific

            start = d.line - 1
            end = min(d.end_line, start + 50)
            source_text = "\n".join(lines[start:end])
            sig = lines[start].strip() if start < len(lines) else d.name
            name = f"{d.class_name}.{d.name}" if d.class_name else d.name

            chunk = CodeChunk(
                id=f"{rel_path}:{name}",
                type=d.kind,
                name=name,
                file=rel_path,
                line=d.line,
                content=f"{sig}\n{d.docstring or ''}\n{source_text}",
                signature=sig,
            )
            if key not in result or d.kind == "method":
                result[key] = chunk

        return list(result.values())

    def _extract_chunks_python(self, file_path: Path) -> List[CodeChunk]:
        """Extract code chunks from a Python file using AST."""
        import ast

        chunks = []
        rel_path = str(file_path.relative_to(self.project_path))

        try:
            content = file_path.read_text()
            tree = ast.parse(content)
        except Exception:
            return chunks

        lines = content.splitlines()

        # Walk full AST to find all definitions (including nested)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                chunk = self._node_to_chunk(node, lines, rel_path, 'function')
                if chunk:
                    chunks.append(chunk)

            elif isinstance(node, ast.AsyncFunctionDef):
                chunk = self._node_to_chunk(node, lines, rel_path, 'function')
                if chunk:
                    chunks.append(chunk)

            elif isinstance(node, ast.ClassDef):
                chunk = self._node_to_chunk(node, lines, rel_path, 'class')
                if chunk:
                    chunks.append(chunk)

                # Also extract methods with class context (more specific)
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_chunk = self._node_to_chunk(
                            item, lines, rel_path, 'method',
                            class_name=node.name
                        )
                        if method_chunk:
                            chunks.append(method_chunk)

        # Dedupe: when same (file, line) appears multiple times, keep the most specific
        # (method > function, since method has class context)
        seen_lines = {}  # (file, line) -> chunk
        for chunk in chunks:
            key = (chunk.file, chunk.line)
            if key not in seen_lines:
                seen_lines[key] = chunk
            elif chunk.type == 'method':
                # Prefer method over function (more specific)
                seen_lines[key] = chunk

        return list(seen_lines.values())

    def _node_to_chunk(
        self,
        node,
        lines: List[str],
        file_path: str,
        chunk_type: str,
        class_name: str = None
    ) -> Optional[CodeChunk]:
        """Convert an AST node to a CodeChunk."""
        import ast  # ensure ast is available in this scope
        try:
            # Get the full source for this node
            start_line = node.lineno - 1
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1

            # Limit chunk size to avoid huge embeddings
            max_lines = 50
            if end_line - start_line > max_lines:
                end_line = start_line + max_lines

            source_lines = lines[start_line:end_line]
            content = "\n".join(source_lines)

            # Build signature
            if chunk_type == 'class':
                signature = f"class {node.name}"
                if node.bases:
                    base_names = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            base_names.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            base_names.append(f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else base.attr)
                    signature += f"({', '.join(base_names)})"
            else:
                # Function/method signature
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    args.append(arg_str)
                signature = f"def {node.name}({', '.join(args)})"
                if node.returns:
                    signature += f" -> {ast.unparse(node.returns)}"

            # Build name and id
            if class_name:
                name = f"{class_name}.{node.name}"
            else:
                name = node.name

            chunk_id = f"{file_path}:{name}"

            # Get docstring if available
            docstring = ast.get_docstring(node) or ""

            # Content for embedding: signature + docstring + code
            embed_content = f"{signature}\n{docstring}\n{content}"

            return CodeChunk(
                id=chunk_id,
                type=chunk_type,
                name=name,
                file=file_path,
                line=node.lineno,
                content=embed_content,
                signature=signature,
            )
        except Exception as e:
            if DEBUG:
                import traceback
                print(f"[RAG] _node_to_chunk error: {type(e).__name__}: {e}", file=sys.stderr)
                traceback.print_exc()
            return None

    def _get_source_files(self) -> List[Path]:
        """Get all indexable source files in the project."""
        # Determine which extensions to look for
        extensions = {".py"}
        try:
            from compass.agents.neo.treesitter import supported_extensions
            extensions.update(supported_extensions())
        except Exception:
            pass

        # Always exclude these patterns (can't be overridden)
        always_exclude = [
            "__pycache__", ".venv", "venv", "node_modules", ".git",
            ".hg", ".svn", "vendor", "target", ".tox", ".mypy_cache",
            "build", "dist",
        ]

        # Load additional exclusions from index config
        index_config = self.compass_dir / "index.json"
        exclude_patterns = always_exclude.copy()

        if index_config.exists():
            try:
                config = json.loads(index_config.read_text())
                for pattern in config.get("exclude", []):
                    if pattern not in exclude_patterns:
                        exclude_patterns.append(pattern)
            except Exception:
                pass

        files = []
        for ext in extensions:
            for src_file in self.project_path.rglob(f"*{ext}"):
                rel_path = str(src_file.relative_to(self.project_path))

                excluded = False
                for pattern in exclude_patterns:
                    if pattern in rel_path:
                        excluded = True
                        break

                if not excluded and src_file.is_file():
                    files.append(src_file)

        return files

    def build_index(self, force: bool = False, background: bool = False) -> int:
        """Build or update the embeddings index.

        Args:
            force: If True, rebuild everything. If False, only update changed files.
            background: If True, run in background thread.

        Returns:
            Number of chunks embedded.
        """
        if background:
            thread = threading.Thread(
                target=self._build_index_safe, args=(force,), daemon=True,
            )
            thread.start()
            return 0  # Return immediately, actual count will be available later
        else:
            return self._build_index_background(force)

    def _build_index_safe(self, force: bool = False):
        """Wrapper for background thread -- swallows errors silently."""
        try:
            self._build_index_background(force)
        except Exception:
            if DEBUG:
                import traceback
                traceback.print_exc()

    def _build_index_background(self, force: bool = False) -> int:
        """Internal method to build index in background."""
        self.compass_dir.mkdir(parents=True, exist_ok=True)

        # Load existing metadata if not forcing rebuild
        file_hashes: Dict[str, str] = {}
        if not force and self.metadata_path.exists():
            try:
                meta = json.loads(self.metadata_path.read_text())
                file_hashes = meta.get("file_hashes", {})
                self._load_embeddings()
            except Exception:
                pass

        # Get current files and their hashes
        current_files = self._get_source_files()
        current_hashes = {
            str(f.relative_to(self.project_path)): self._get_file_hash(f)
            for f in current_files
        }

        # Detect missing files
        missing_files = set(file_hashes.keys()) - set(current_hashes.keys())
        MISSING_FILE_THRESHOLD = 5
        if force:
            # Reset counter on explicit rebuild
            self.missing_file_counter = 0
        if missing_files:
            self.missing_file_counter += len(missing_files)

        if not force:
            # Check if missing file counter exceeds threshold to trigger full rebuild

            self._prune_stale_entries()

        if self.missing_file_counter > MISSING_FILE_THRESHOLD:
            # Too many missing files across builds, rebuild entire index
            force = True
            # Reset file_hashes and chunks
            file_hashes = {}
            self.chunks = {}
            # Reset counter after rebuild
            self.missing_file_counter = 0
            # Rebuild index
            return self._build_index_background(force=True)

        # Determine which files changed
        changed_files = []
        for f in current_files:
            rel_path = str(f.relative_to(self.project_path))
            if rel_path not in file_hashes or file_hashes[rel_path] != current_hashes[rel_path]:
                changed_files.append(f)

        # Increment counter for missing files and prune stale entries


        # Extract chunks from changed files
        new_chunks = []
        for f in changed_files:
            new_chunks.extend(self._extract_chunks_from_file(f))

        if not new_chunks and self.embeddings is not None:
            # No changes, existing index is valid
            return len(self.chunk_ids)

        # If incremental update, remove old chunks from changed files
        if not force and self.chunks:
            changed_rel_paths = {str(f.relative_to(self.project_path)) for f in changed_files}
            self.chunks = {
                k: v for k, v in self.chunks.items()
                if v.file not in changed_rel_paths
            }

        # Add new chunks
        for chunk in new_chunks:
            self.chunks[chunk.id] = chunk

        # Embed chunks (incremental when possible)
        if self.chunks:
            changed_ids = {chunk.id for chunk in new_chunks} if not force and new_chunks else None
            self._embed_chunks(changed_chunk_ids=changed_ids)

        # Save
        self._save_embeddings(current_hashes)
        self._prune_stale_entries()

        return len(self.chunks)



    def _embed_chunks(self, changed_chunk_ids: Optional[Set[str]] = None):
        """Embed chunks, reusing existing embeddings for unchanged chunks.

        Args:
            changed_chunk_ids: When provided, only embed these chunks (plus any
                chunks not yet in the embedding matrix). Reuse existing vectors
                for everything else.
        """
        sorted_ids = sorted(self.chunks.keys())

        # Build lookup of existing chunk_id -> embedding vector
        old_lookup: Dict[str, np.ndarray] = {}
        if changed_chunk_ids is not None and self.embeddings is not None and self.chunk_ids:
            for idx, cid in enumerate(self.chunk_ids):
                if idx < len(self.embeddings):
                    old_lookup[cid] = self.embeddings[idx]

        # Determine which chunks actually need embedding
        need_embed = []
        for cid in sorted_ids:
            if changed_chunk_ids is not None and cid not in changed_chunk_ids and cid in old_lookup:
                continue  # reuse existing
            need_embed.append(cid)

        if need_embed:
            texts = [self.chunks[cid].content for cid in need_embed]
            new_vectors = _embed_texts(texts)
            if new_vectors is None:
                return  # no backend -- skip embedding

            # Normalize
            norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            new_vectors = new_vectors / norms

            # Merge into lookup
            for i, cid in enumerate(need_embed):
                old_lookup[cid] = new_vectors[i]

        # Reassemble in sorted order
        self.chunk_ids = sorted_ids
        if old_lookup:
            self.embeddings = np.array([old_lookup[cid] for cid in sorted_ids])
        else:
            # Everything is new, embed all
            texts = [self.chunks[cid].content for cid in sorted_ids]
            result = _embed_texts(texts)
            if result is None:
                return  # no backend -- skip embedding
            self.embeddings = result
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.embeddings = self.embeddings / norms

    def _save_embeddings(self, file_hashes: Dict[str, str]):
        """Save embeddings and metadata to disk."""
        if self.embeddings is None:
            return

        # Save embeddings as numpy
        np.savez_compressed(
            self.embeddings_path,
            embeddings=self.embeddings,
            chunk_ids=np.array(self.chunk_ids, dtype=object),
        )

        # Save metadata as JSON
        meta = {
            "file_hashes": file_hashes,
            "chunks": {cid: asdict(chunk) for cid, chunk in self.chunks.items()},
        }
        self.metadata_path.write_text(json.dumps(meta, indent=2))

    def _load_embeddings(self) -> bool:
        """Load embeddings from disk. Validates consistency."""
        if not self.embeddings_path.exists() or not self.metadata_path.exists():
            return False

        try:
            # Load numpy arrays
            data = np.load(self.embeddings_path, allow_pickle=True)
            self.embeddings = data["embeddings"]
            self.chunk_ids = list(data["chunk_ids"])

            # Load metadata
            meta = json.loads(self.metadata_path.read_text())
            self.chunks = {
                cid: CodeChunk(**chunk_data)
                for cid, chunk_data in meta.get("chunks", {}).items()
            }

            # Validate consistency -- all three must agree
            if len(self.chunk_ids) != len(self.embeddings):
                self.embeddings = None
                self.chunk_ids = []
                self.chunks = {}
                return False

            return True
        except Exception:
            return False

    def _prune_stale_entries(self) -> Set[str]:
        """Remove entries for deleted files from the index and return removed IDs."""
        removed_ids = set()
        for cid, chunk in list(self.chunks.items()):
            file_path = self.project_path / chunk.file
            if not file_path.exists():
                removed_ids.add(cid)
                del self.chunks[cid]
        if not removed_ids:
            return removed_ids
        # Keep mask: True for chunk_ids to retain
        keep_mask = [cid not in removed_ids for cid in self.chunk_ids]
        self.chunk_ids = [cid for cid, keep in zip(self.chunk_ids, keep_mask) if keep]
        # Shrink embeddings matrix to match
        if self.embeddings is not None and len(keep_mask) == len(self.embeddings):
            self.embeddings = self.embeddings[keep_mask]
        return removed_ids
    def _build_full_index(self, force: bool = True) -> int:
        """Trigger a full rebuild of the index."""
        # Reset counter before full rebuild
        self.missing_file_counter = 0
        # Perform full rebuild
        return self.build_index(force=True)



class CodeRetriever:
    """Retrieves relevant code chunks based on query similarity."""

    def __init__(self, embedder: CodeEmbedder):
        self.embedder = embedder

    def query(self, query: str, top_k: int = 10) -> List[Tuple[CodeChunk, float]]:
        """Find the most relevant code chunks for a query.

        Args:
            query: Natural language query or code snippet
            top_k: Number of results to return

        Returns:
            List of (chunk, score) tuples, sorted by relevance
        """
        if self.embedder.embeddings is None or len(self.embedder.chunk_ids) == 0:
            return []

        # Embed query using available model
        query_embedding = _embed_query(query)
        if query_embedding is None:
            return []

        # Cosine similarity via dot product (embeddings are normalized)
        scores = np.dot(self.embedder.embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk_id = self.embedder.chunk_ids[idx]
            chunk = self.embedder.chunks.get(chunk_id)
            if chunk:
                results.append((chunk, float(scores[idx])))

        return results

    def format_context(self, results: List[Tuple[CodeChunk, float]], max_chars: int = 8000) -> str:
        """Format retrieval results as context string for the Planner.

        Args:
            results: List of (chunk, score) from query()
            max_chars: Maximum characters to include

        Returns:
            Formatted string with relevant code snippets
        """
        if not results:
            return ""

        lines = ["--- RELEVANT CODE (from RAG) ---"]
        current_chars = len(lines[0])

        for chunk, score in results:
            # Format: file:line (score) - signature
            header = f"\n# {chunk.file}:{chunk.line} ({score:.2f}) - {chunk.type}"
            entry = f"{header}\n{chunk.signature}"

            # Add docstring snippet if available
            content_lines = chunk.content.split("\n")
            docstring_lines = []
            in_docstring = False
            for line in content_lines[1:10]:  # Skip signature, check first few lines
                if '"""' in line or "'''" in line:
                    in_docstring = not in_docstring
                    docstring_lines.append(line)
                elif in_docstring:
                    docstring_lines.append(line)

            if docstring_lines:
                entry += "\n" + "\n".join(docstring_lines[:5])  # First 5 lines of docstring

            if current_chars + len(entry) > max_chars:
                break

            lines.append(entry)
            current_chars += len(entry)

        return "\n".join(lines)


# Convenience function for one-shot retrieval
def get_relevant_context(project_path: str, query: str, top_k: int = 10) -> RAGResult:
    """Retrieve relevant context from the in-memory index.

    Pure query -- never triggers indexing. If nothing is indexed, returns
    a gap result. Use `reindex()` or the CLI (`python -m compass.agents.neo.rag rebuild`)
    to populate the index explicitly.

    Args:
        project_path: Path to the project root
        query: The user's query or task description
        top_k: Number of chunks to retrieve

    Returns:
        RAGResult with context string and score metadata for gap detection
    """
    return retrieve_cached(project_path, query, top_k)


def retrieve_cached(project_path: str, query: str, top_k: int = 10) -> RAGResult:
    """Retrieve relevant context from cached index without rebuilding.

    Args:
        project_path: Path to the project root
        query: The user's query or task description
        top_k: Number of chunks to retrieve

    Returns:
        RAGResult with context string and score metadata for gap detection
    """
    embedder = get_embedder(project_path)

    if embedder.embeddings is None:
        return RAGResult(
            context="",
            max_score=0.0,
            avg_score=0.0,
            has_gap=True,
            top_matches=[]
        )

    retriever = CodeRetriever(embedder)
    results = retriever.query(query, top_k=top_k)
    context = retriever.format_context(results)

    # Compute score metrics for gap detection
    if results:
        scores = [score for _, score in results]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        top_matches = [(chunk.id, score) for chunk, score in results[:5]]
    else:
        max_score = 0.0
        avg_score = 0.0
        top_matches = []

    has_gap = max_score < GAP_THRESHOLD

    return RAGResult(
        context=context,
        max_score=max_score,
        avg_score=avg_score,
        has_gap=has_gap,
        top_matches=top_matches
    )


def reindex(project_path: str, force: bool = False) -> int:
    """Rebuild the index for the project.

    Args:
        project_path: Path to the project root
        force: If True, rebuild everything. If False, only update changed files.

    Returns:
        Number of chunks embedded.
    """
    embedder = get_embedder(project_path)
    return embedder.build_index(force=force)


def reindex_in_background(project_path: str, force: bool = False) -> None:
    """Rebuild the index for the project in a background thread.

    Args:
        project_path: Path to the project root
        force: If True, rebuild everything. If False, only update changed files.
    """
    embedder = get_embedder(project_path)
    embedder.build_index(force=force, background=True)


# =============================================================================
# CLI: python -m compass.agents.neo.rag <command> [options]
# =============================================================================

def _cli():
    """RAG index management.

    Usage:
        python -m compass.agents.neo.rag rebuild [--force] [path]
        python -m compass.agents.neo.rag status [path]
        python -m compass.agents.neo.rag query <text> [path]
    """
    import time

    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("Usage: python -m compass.agents.neo.rag <command> [options]")
        print()
        print("Commands:")
        print("  rebuild [--force] [path]   Build/update the index (default: .)")
        print("  status [path]              Show index stats")
        print("  query <text> [path]        Search the index")
        return

    cmd = args[0]
    rest = args[1:]

    force = "--force" in rest
    rest = [a for a in rest if a != "--force"]

    if cmd == "rebuild":
        project = rest[0] if rest else "."
        print(f"Indexing {Path(project).resolve()} {'(full rebuild)' if force else '(incremental)'}...")
        t0 = time.perf_counter()
        count = reindex(project, force=force)
        elapsed = time.perf_counter() - t0
        print(f"Done: {count} chunks in {elapsed:.1f}s")

    elif cmd == "status":
        project = rest[0] if rest else "."
        embedder = get_embedder(project)
        n_chunks = len(embedder.chunks)
        n_embeds = len(embedder.embeddings) if embedder.embeddings is not None else 0
        dim = embedder.embeddings.shape[1] if embedder.embeddings is not None and embedder.embeddings.ndim == 2 else 0
        has_disk = embedder.metadata_path.exists()
        print(f"Project:    {Path(project).resolve()}")
        print(f"Chunks:     {n_chunks}")
        print(f"Embeddings: {n_embeds} x {dim}-dim")
        print(f"On disk:    {'yes' if has_disk else 'no'}")
        if n_chunks:
            files = {c.file for c in embedder.chunks.values()}
            print(f"Files:      {len(files)}")

    elif cmd == "query":
        if not rest:
            print("Usage: python -m compass.agents.neo.rag query <text> [path]")
            return
        text = rest[0]
        project = rest[1] if len(rest) > 1 else "."
        result = get_relevant_context(project, text, top_k=5)
        print(f"max_score: {result.max_score:.3f}  has_gap: {result.has_gap}")
        if result.context:
            print(result.context)
        else:
            print("(no results -- index may be empty, run: rebuild)")

    else:
        print(f"Unknown command: {cmd}")
        print("Run with --help for usage")


if __name__ == "__main__":
    _cli()
