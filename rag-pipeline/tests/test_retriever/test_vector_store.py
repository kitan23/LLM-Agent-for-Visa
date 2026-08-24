"""
Tests for the FAISS vector store management.
"""

import pytest
from unittest.mock import patch

from langchain_community.vectorstores import FAISS

from src.retriever import vector_store as vs
from src.retriever.vector_store import (
    DUMMY_CHUNK,
    EMBEDDING_MODEL_NAME,
    append_to_vector_store,
    compute_document_hash,
    create_or_load_vector_store,
    load_vector_store,
    store_corpus_texts,
)


class TestComputeDocumentHash:
    """Tests for content hashing."""

    def test_deterministic(self):
        chunks = ["alpha", "beta"]
        h1 = compute_document_hash(chunks, "model")
        h2 = compute_document_hash(chunks, "model")
        assert h1 == h2

    def test_changes_with_content(self):
        h1 = compute_document_hash(["alpha"], "model")
        h2 = compute_document_hash(["beta"], "model")
        assert h1 != h2

    def test_changes_with_model(self):
        h1 = compute_document_hash(["alpha"], "model-a")
        h2 = compute_document_hash(["alpha"], "model-b")
        assert h1 != h2

    def test_truncated_length(self):
        h = compute_document_hash(["alpha"], "model")
        assert len(h) == 16


class TestStoreCorpusTexts:
    """Tests for corpus extraction."""

    def test_excludes_dummy_chunks(self, fake_embeddings):
        store = FAISS.from_texts([DUMMY_CHUNK, "real text"], embedding=fake_embeddings)
        texts = store_corpus_texts(store)
        assert texts == ["real text"]

    def test_empty_store_returns_empty(self, fake_embeddings):
        store = FAISS.from_texts([DUMMY_CHUNK], embedding=fake_embeddings)
        assert store_corpus_texts(store) == []


class TestCreateOrLoadVectorStore:
    """Tests for cache-aware store creation."""

    def test_creates_and_persists(self, tmp_path, fake_embeddings):
        with patch.object(vs, "get_embeddings", return_value=fake_embeddings):
            store = create_or_load_vector_store(
                chunks=["hello world", "visa rules"],
                embeddings=fake_embeddings,
                cache_dir=str(tmp_path),
            )

        assert store.index.ntotal == 2
        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "content_hash.txt").exists()

    def test_cache_hit_avoids_rebuild(self, tmp_path, fake_embeddings):
        with patch.object(vs, "get_embeddings", return_value=fake_embeddings), \
             patch.object(vs.FAISS, "from_texts", wraps=FAISS.from_texts) as mock_from:
            create_or_load_vector_store(
                chunks=["hello world"], embeddings=fake_embeddings, cache_dir=str(tmp_path)
            )
            assert mock_from.call_count == 1

            # Same content: should load from cache, not rebuild
            store = create_or_load_vector_store(
                chunks=["hello world"], embeddings=fake_embeddings, cache_dir=str(tmp_path)
            )
            assert mock_from.call_count == 1
            assert store.index.ntotal == 1

    def test_cache_invalidated_on_content_change(self, tmp_path, fake_embeddings):
        with patch.object(vs, "get_embeddings", return_value=fake_embeddings), \
             patch.object(vs.FAISS, "from_texts", wraps=FAISS.from_texts) as mock_from:
            create_or_load_vector_store(
                chunks=["hello world"], embeddings=fake_embeddings, cache_dir=str(tmp_path)
            )
            create_or_load_vector_store(
                chunks=["different content"], embeddings=fake_embeddings, cache_dir=str(tmp_path)
            )
            assert mock_from.call_count == 2

    def test_cache_invalidated_on_model_change(self, tmp_path, fake_embeddings):
        with patch.object(vs, "get_embeddings", return_value=fake_embeddings):
            create_or_load_vector_store(
                chunks=["hello world"], embeddings=fake_embeddings, cache_dir=str(tmp_path)
            )

        with patch.object(vs, "EMBEDDING_MODEL_NAME", "other-model"), \
             patch.object(vs.FAISS, "from_texts", wraps=FAISS.from_texts) as mock_from:
            create_or_load_vector_store(
                chunks=["hello world"], embeddings=fake_embeddings, cache_dir=str(tmp_path)
            )
            assert mock_from.call_count == 1


class TestAppendToVectorStore:
    """Tests for merging new chunks into an existing knowledge base."""

    def test_append_preserves_existing_content(self, tmp_path, fake_embeddings):
        with patch.object(vs, "get_embeddings", return_value=fake_embeddings):
            existing = FAISS.from_texts(["existing doc text"], embedding=fake_embeddings)

            merged = append_to_vector_store(
                existing_store=existing,
                new_chunks=["new doc text"],
                vector_store_path=tmp_path,
            )

        assert merged.index.ntotal == 2
        results = merged.similarity_search("new doc text", k=2)
        contents = {r.page_content for r in results}
        assert "new doc text" in contents

    def test_append_persists_merged_store(self, tmp_path, fake_embeddings):
        with patch.object(vs, "get_embeddings", return_value=fake_embeddings):
            existing = FAISS.from_texts(["first"], embedding=fake_embeddings)
            append_to_vector_store(existing, ["second"], tmp_path)

            reloaded = load_vector_store(tmp_path)
        assert reloaded.index.ntotal == 2

    def test_first_append_on_dummy_store(self, tmp_path, fake_embeddings):
        """A placeholder-only store gets replaced by real content."""
        with patch.object(vs, "get_embeddings", return_value=fake_embeddings):
            dummy = FAISS.from_texts([DUMMY_CHUNK], embedding=fake_embeddings)

            merged = append_to_vector_store(dummy, ["real chunk"], tmp_path)

        assert merged.index.ntotal == 1
        assert store_corpus_texts(merged) == ["real chunk"]

    def test_writes_content_hash_of_merged_corpus(self, tmp_path, fake_embeddings):
        with patch.object(vs, "get_embeddings", return_value=fake_embeddings):
            existing = FAISS.from_texts(["aaa"], embedding=fake_embeddings)
            append_to_vector_store(existing, ["bbb"], tmp_path)

        expected = compute_document_hash(["aaa", "bbb"], EMBEDDING_MODEL_NAME)
        assert (tmp_path / "content_hash.txt").read_text().strip() == expected


class TestLoadVectorStore:
    """Tests for loading persisted stores."""

    def test_missing_path_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_vector_store(tmp_path / "does-not-exist")


class TestBuildVectorStore:
    """Tests for the build entrypoint."""

    def test_builds_with_chunks(self, tmp_path, fake_embeddings):
        with patch.object(vs, "get_embeddings", return_value=fake_embeddings):
            store = vs.build_vector_store(["chunk one"], str(tmp_path))
        assert store.index.ntotal == 1

    def test_build_with_empty_uses_dummy(self, tmp_path, fake_embeddings):
        with patch.object(vs, "get_embeddings", return_value=fake_embeddings):
            store = vs.build_vector_store([], str(tmp_path))
        assert store.index.ntotal == 1
