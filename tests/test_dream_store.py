"""Tests for DreamStore -- the shared dream transcript engine."""

import json
import numpy as np
import pytest
from pathlib import Path

from compass.generators._transcript import DreamStore


@pytest.fixture
def store(tmp_path):
    """DreamStore with temporary directories."""
    transcripts = tmp_path / "transcripts"
    index = tmp_path / ".index"
    return DreamStore(transcripts, index)


@pytest.fixture
def neo_entries():
    """Screen automation dream entries."""
    return [
        {"said": "open linkedin and message a contact"},
        {"did": [
            {"action": "navigate", "url": "https://linkedin.com"},
            {"action": "click_text", "target": "Messaging"},
            {"action": "type", "value": "Hello there"},
        ]},
    ]


@pytest.fixture
def trinity_entries():
    """Code/research dream entries."""
    return [
        {"said": "what is the revenue trend for Q4"},
        {"did": [
            {"step": "analyze_revenue", "description": "compute quarterly revenue",
             "value": "Q4 up 15% YoY", "fact_type": "text"},
            {"step": "compare_quarters", "description": "compare Q3 vs Q4",
             "value": "Q4 beat Q3 by 8%", "fact_type": "text"},
        ]},
    ]


class TestSaveLoad:

    def test_save_creates_file(self, store, neo_entries):
        path = store.save("linkedin_dm", neo_entries, target="Safari")
        assert path.exists()
        assert path.suffix == ".yaml"

    def test_load_roundtrip(self, store, neo_entries):
        store.save("linkedin_dm", neo_entries, target="Safari")
        loaded = store.load("linkedin_dm")
        assert loaded["name"] == "linkedin_dm"
        assert loaded["target"] == "Safari"
        assert len(loaded["entries"]) == 2
        assert loaded["entries"][0]["said"] == "open linkedin and message a contact"
        assert loaded["entries"][1]["did"][0]["action"] == "navigate"

    def test_load_nonexistent(self, store):
        assert store.load("does_not_exist") is None

    def test_list_names(self, store, neo_entries, trinity_entries):
        store.save("alpha", neo_entries)
        store.save("beta", trinity_entries)
        names = store.list_names()
        assert "alpha" in names
        assert "beta" in names

    def test_list_empty(self, store):
        assert store.list_names() == []

    def test_save_with_meta(self, store, trinity_entries):
        store.save("revenue", trinity_entries, domain="trinity", workspace="/tmp")
        loaded = store.load("revenue")
        assert loaded["domain"] == "trinity"
        assert loaded["workspace"] == "/tmp"


class TestKeywordSearch:
    """Search falls back to keyword overlap when embeddings are unavailable."""

    def test_keyword_match(self, store, neo_entries):
        store.save("linkedin_dm", neo_entries, target="Safari")
        results = store.search("message on linkedin")
        assert len(results) >= 1
        assert results[0][0] == "linkedin_dm"

    def test_keyword_no_match(self, store, neo_entries, monkeypatch):
        store.save("linkedin_dm", neo_entries)
        # Force keyword-only by breaking embeddings
        import compass.generators._transcript as mod
        monkeypatch.setattr(mod, "_embed_texts", lambda t: (_ for _ in ()).throw(RuntimeError("no embeddings")))
        results = store.search("fibonacci sequence")
        assert len(results) == 0

    def test_keyword_ranking(self, store, neo_entries, trinity_entries):
        store.save("linkedin_dm", neo_entries)
        store.save("revenue_analysis", trinity_entries)
        results = store.search("revenue quarterly trend")
        assert len(results) >= 1
        assert results[0][0] == "revenue_analysis"

    def test_returns_transcript(self, store, neo_entries):
        store.save("linkedin_dm", neo_entries)
        results = store.search("linkedin")
        assert len(results) == 1
        name, score, transcript = results[0]
        assert transcript["entries"][0]["said"] == "open linkedin and message a contact"


class TestSemanticSearch:
    """Search with monkeypatched embeddings."""

    @pytest.fixture
    def seeded_store(self, store, neo_entries, trinity_entries, monkeypatch):
        store.save("linkedin_dm", neo_entries, target="Safari")
        store.save("revenue_analysis", trinity_entries, domain="trinity")

        # Fake embeddings: linkedin_dm -> [1,0], revenue -> [0,1]
        def fake_embed(texts):
            vecs = []
            for t in texts:
                if "linkedin" in t.lower():
                    vecs.append([1.0, 0.0])
                else:
                    vecs.append([0.0, 1.0])
            return np.array(vecs)

        def fake_query(query):
            if "message" in query.lower() or "linkedin" in query.lower():
                return np.array([1.0, 0.0])
            return np.array([0.0, 1.0])

        import compass.generators._transcript as mod
        monkeypatch.setattr(mod, "_embed_texts", fake_embed)
        monkeypatch.setattr(mod, "_embed_query", fake_query)
        return store

    def test_semantic_finds_linkedin(self, seeded_store):
        results = seeded_store.search("message someone")
        assert len(results) >= 1
        assert results[0][0] == "linkedin_dm"

    def test_semantic_finds_revenue(self, seeded_store):
        results = seeded_store.search("quarterly earnings")
        assert len(results) >= 1
        assert results[0][0] == "revenue_analysis"

    def test_build_index_returns_count(self, seeded_store):
        n = seeded_store.build_index()
        assert n == 2

    def test_build_index_caches(self, seeded_store):
        seeded_store.build_index()
        # Second call should short-circuit (same hash)
        n = seeded_store.build_index()
        assert n == 2

    def test_build_index_force_rebuilds(self, seeded_store):
        seeded_store.build_index()
        n = seeded_store.build_index(force=True)
        assert n == 2

    def test_cache_persists_to_disk(self, seeded_store):
        seeded_store.build_index()
        assert seeded_store._cache_path().exists()
        assert seeded_store._meta_path().exists()

        meta = json.loads(seeded_store._meta_path().read_text())
        assert "hash" in meta
        assert len(meta["names"]) == 2

    def test_cache_loads_from_disk(self, seeded_store):
        seeded_store.build_index()

        # Create a fresh store pointing at same dirs
        fresh = DreamStore(seeded_store.transcripts_dir, seeded_store.index_dir)
        n = fresh.build_index()
        assert n == 2
        assert fresh._embeddings is not None


class TestContext:

    def test_neo_context(self, store, neo_entries):
        store.save("linkedin_dm", neo_entries, target="Safari")
        t = store.load("linkedin_dm")
        section = store.to_context(t)
        assert section.heading == "Past Experience: linkedin_dm"
        assert "Human: open linkedin and message a contact" in section.content
        assert "navigate: https://linkedin.com" in section.content
        assert "click_text: Messaging" in section.content

    def test_trinity_context(self, store, trinity_entries):
        store.save("revenue", trinity_entries, domain="trinity")
        t = store.load("revenue")
        section = store.to_context(t)
        assert "Human: what is the revenue trend for Q4" in section.content
        assert "analyze_revenue" in section.content
        assert "Q4 up 15% YoY" in section.content

    def test_search_and_inject(self, store, neo_entries, monkeypatch):
        store.save("linkedin_dm", neo_entries)

        def fake_embed(texts):
            return np.array([[1.0, 0.0]] * len(texts))

        def fake_query(query):
            return np.array([1.0, 0.0])

        import compass.generators._transcript as mod
        monkeypatch.setattr(mod, "_embed_texts", fake_embed)
        monkeypatch.setattr(mod, "_embed_query", fake_query)

        sections = store.search_and_inject("message someone")
        assert len(sections) >= 1
        assert sections[0].heading.startswith("Past Experience:")
