"""
Integration tests for the FastAPI application.

These tests exercise the HTTP surface of the backend using the FastAPI
TestClient with the assistant and observability layers mocked out.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class FakeAssistant:
    """Stand-in for OPTRagAssistant with configurable behavior."""

    def __init__(self, configured=True, **kwargs):
        self.is_configured = configured
        self.vector_store = MagicMock()
        self.vector_store.index.ntotal = 42
        self.added_files = []

    async def answer_question(self, query, trace=None, generation=None):
        return {
            "answer": f"Answer to: {query}",
            "sources": [{"content": "chunk", "metadata": {"source": "doc.pdf"}}],
            "metadata": {"response_time": 0.01, "model": "fake-model"},
        }

    async def add_documents(self, file_path=None, document_type=None):
        self.added_files.extend(file_path or [])
        return {
            "status": "success",
            "document_count": len(file_path or []),
            "chunk_count": 2,
            "total_vectors": 44,
            "processing_time": 0.02,
        }

    async def astream_response(self, query, cancel_event=None, trace=None, generation=None):
        for token in ["Hello ", "from ", query]:
            yield token

    def list_documents(self):
        return {
            "status": "success",
            "document_count": 1,
            "total_chunks": 3,
            "documents": [{"source": "doc.pdf", "document_type": "immigration", "chunk_count": 3}],
        }


@pytest.fixture
def client():
    """TestClient wired to a fake assistant."""
    with patch("src.main.OPTRagAssistant", FakeAssistant), \
         patch("src.main.langfuse", MagicMock()):
        from src.main import app

        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def unconfigured_client():
    """TestClient wired to an assistant without any LLM provider."""
    with patch("src.main.OPTRagAssistant", lambda **kw: FakeAssistant(configured=False)), \
         patch("src.main.langfuse", MagicMock()):
        from src.main import app

        with TestClient(app) as test_client:
            yield test_client


def parse_sse(text):
    """Parse an SSE payload into a list of data strings."""
    events = []
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                events.append(line[5:].strip())
    return events


class TestBasicEndpoints:
    """Tests for basic service endpoints."""

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["message"] == "OPT-RAG API is running"

    def test_health_healthy(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["vector_vectors"] == 42

    def test_health_degraded_without_provider(self, unconfigured_client):
        resp = unconfigured_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert "LLM" in body["reason"]

    def test_health_under_api_prefix(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestQueryEndpoints:
    """Tests for standard query endpoints."""

    def test_get_query(self, client):
        resp = client.get("/query", params={"q": "What is OPT?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Answer to: What is OPT?"
        assert len(body["sources"]) == 1

    def test_post_query(self, client):
        resp = client.post("/api/query", json={"question": "What is CPT?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"].startswith("Answer to:")
        assert "processing_time" in body

    def test_query_missing_param_rejected(self, client):
        resp = client.get("/query")
        assert resp.status_code == 422

    def test_query_when_unconfigured_returns_error(self, unconfigured_client):
        resp = unconfigured_client.post("/api/query", json={"question": "hi"})
        assert resp.status_code == 200
        body = resp.json()
        assert "error" in body
        assert "RUNPOD_API_KEY" in body["error"]

    def test_get_query_when_unconfigured_returns_error(self, unconfigured_client):
        resp = unconfigured_client.get("/query", params={"q": "hi"})
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_post_query_returns_503_when_llm_fails(self, client):
        """LLM provider failures surface as structured 503 responses."""
        with patch("src.main.assistant") as mock_assistant:
            mock_assistant.is_configured = True
            mock_assistant.answer_question = AsyncMock(side_effect=RuntimeError("providers down"))

            resp = client.post("/api/query", json={"question": "hi"})

        assert resp.status_code == 503
        assert "providers down" in resp.json()["error"]


class TestStreamingEndpoints:
    """Tests for the SSE streaming endpoints."""

    def test_post_stream_sse_contract(self, client):
        resp = client.post(
            "/api/query/stream",
            json={"question": "visa"},
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = parse_sse(resp.text)

        # First event carries the request id
        first = json.loads(events[0])
        assert "request_id" in first

        # Token events are JSON objects with a token field
        tokens = [json.loads(e)["token"] for e in events[1:] if e != "[DONE]"]
        assert tokens == ["Hello ", "from ", "visa"]

        # Stream terminates with the DONE sentinel
        assert events[-1] == "[DONE]"

    def test_get_stream_sse_contract(self, client):
        resp = client.get("/stream", params={"q": "opt"}, headers={"Accept": "text/event-stream"})
        assert resp.status_code == 200

        events = parse_sse(resp.text)
        tokens = [json.loads(e)["token"] for e in events if e not in ("[DONE]",)]
        assert "".join(tokens) == "Hello from opt"
        assert events[-1] == "[DONE]"

    def test_post_stream_unconfigured_emits_error_event(self, unconfigured_client):
        resp = unconfigured_client.post(
            "/api/query/stream",
            json={"question": "visa"},
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        events = parse_sse(resp.text)
        error_event = json.loads(events[0])
        assert "error" in error_event
        assert events[-1] == "[DONE]"

    def test_cancel_unknown_request_id(self, client):
        resp = client.post("/api/cancel", json={"request_id": "nope"})
        assert resp.status_code == 200
        assert resp.json()["cancelled"] is False

    def test_cancel_via_path_param(self, client):
        resp = client.post("/cancel/unknown-id")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_found"


class TestCancellation:
    """Tests for generation cancellation."""

    def test_cancel_active_generation(self, client, monkeypatch):
        from threading import Event
        from src.main import active_generations

        event = Event()
        monkeypatch.setitem(active_generations, "active-1", event)

        resp = client.post("/api/cancel", json={"request_id": "active-1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["cancelled"] is True
        assert event.is_set()

    def test_metrics_summary(self, client):
        resp = client.get("/metrics/summary")
        assert resp.status_code == 200
        assert "Langfuse" in resp.json()["status"]


class TestDocumentEndpoints:
    """Tests for document management endpoints."""

    def test_list_documents(self, client):
        resp = client.get("/documents")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["document_count"] == 1

    def test_upload_text_document(self, client, tmp_path, monkeypatch):
        # Keep temp artifacts inside the test's tmp dir
        monkeypatch.chdir(tmp_path)

        content = b"F-1 visa holders may remain in the US for 60 days after completion."
        resp = client.post(
            "/documents",
            files={"file": ("rules.txt", content, "text/plain")},
            data={"document_type": "immigration"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["chunk_count"] == 2

        # The temporary upload file must be cleaned up afterwards
        assert not (tmp_path / "temp" / "rules.txt").exists()
