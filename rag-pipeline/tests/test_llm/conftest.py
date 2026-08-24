"""
Test fixtures for OPT-RAG Assistant tests.
"""

import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document


@pytest.fixture
def mock_vector_store():
    """Mock FAISS vector store returning deterministic search results."""
    store = MagicMock()
    store.index.ntotal = 100
    store.similarity_search.return_value = [
        Document(
            page_content="OPT requires filing Form I-765.",
            metadata={"source": "opt_guide.pdf", "document_type": "immigration"},
        ),
        Document(
            page_content="F-1 students may apply up to 90 days before program completion.",
            metadata={"source": "visa_faq.pdf", "document_type": "immigration"},
        ),
        Document(
            page_content="CPT must be authorized before employment begins.",
            metadata={"source": "work_auth.pdf", "document_type": "immigration"},
        ),
    ]
    return store


@pytest.fixture
def runpod_env(monkeypatch):
    """Environment configured with RunPod credentials."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "rp-endpoint")


@pytest.fixture
def openai_env(monkeypatch):
    """Environment configured with only OpenAI credentials."""
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")


@pytest.fixture
def mock_langfuse_objects():
    """No-op trace/generation doubles matching the Langfuse surface used."""
    class FakeGeneration:
        def __init__(self):
            self.calls = []

        def end(self, **kwargs):
            self.calls.append(kwargs)

    class FakeTrace:
        def __init__(self):
            self.updates = []
            self.gen = FakeGeneration()

        def update(self, **kwargs):
            self.updates.append(kwargs)

        def generation(self, *args, **kwargs):
            return self.gen

    trace = FakeTrace()
    return trace, trace.gen
