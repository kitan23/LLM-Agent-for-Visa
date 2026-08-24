"""
Unit tests for the OPT-RAG Assistant class.

These tests verify the functionality of the OPTRagAssistant class using mock
objects and fixtures. For interactive testing with real models, use the
manual_test_assistant.py script in the project root.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.assistant import OPTRagAssistant


def _completion(text: str):
    """Build a fake non-streaming chat completion."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def make_stream_chunks(contents):
    """Build fake streaming chunks; an empty choices list is included last."""
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=c))])
        for c in contents
    ]
    # Simulate the final chunk some providers send with no choices
    chunks.append(SimpleNamespace(choices=[]))
    return chunks


async def _async_iter(items):
    for item in items:
        yield item


def build_assistant(mock_vector_store, provider="runpod"):
    """Create an OPTRagAssistant with external effects patched out."""
    vector_store_path = "/tmp/test-vector-store"

    settings = SimpleNamespace(
        runpod_api_key="rp-key" if provider in ("runpod", "both") else None,
        runpod_endpoint_id="rp-ep" if provider in ("runpod", "both") else None,
        runpod_model_name="test-runpod-model",
        openai_api_key="oai-key" if provider in ("openai", "both") else None,
        openai_model_name="test-openai-model",
        vector_store_path=vector_store_path,
    )

    with patch("src.llm.assistant.get_settings", return_value=settings), \
         patch("src.llm.assistant.load_vector_store", return_value=mock_vector_store), \
         patch("src.llm.assistant.AsyncOpenAI") as mock_client_cls:
        assistant = OPTRagAssistant(vector_store_path=vector_store_path)

    return assistant, mock_client_cls


class TestInit:
    """Tests for assistant initialization."""

    def test_runpod_provider(self, mock_vector_store):
        assistant, client_cls = build_assistant(mock_vector_store, provider="runpod")

        assert assistant.is_configured is True
        assert assistant.client is not None
        assert assistant.fallback_client is None
        assert "runpod.ai" in str(client_cls.call_args)

    def test_openai_only_fallback(self, mock_vector_store):
        assistant, _ = build_assistant(mock_vector_store, provider="openai")

        assert assistant.is_configured is True
        assert assistant.client is None
        assert assistant.fallback_client is not None

    def test_no_providers_degrades_gracefully(self, mock_vector_store):
        assistant, _ = build_assistant(mock_vector_store, provider=None)

        assert assistant.is_configured is False
        assert assistant.client is None
        assert assistant.fallback_client is None
        # Vector store still loaded so document management works
        assert assistant.vector_store is mock_vector_store


class TestMessages:
    """Tests for prompt construction."""

    def test_message_structure(self, mock_vector_store):
        assistant, _ = build_assistant(mock_vector_store)
        messages = assistant._create_messages(context="CTX", question="Q?")

        assert messages[0]["role"] == "system"
        assert "visa" in messages[0]["content"].lower()
        assert "CTX" in messages[1]["content"]
        assert "Q?" in messages[1]["content"]
        assert messages[1]["role"] == "user"

    def test_clean_response(self, mock_vector_store):
        assistant, _ = build_assistant(mock_vector_store)
        assert assistant._clean_response("answer</s>") == "answer"


class TestAnswerQuestion:
    """Tests for the standard (non-streaming) RAG flow."""

    @pytest.mark.asyncio
    async def test_answer_question_happy_path(self, mock_vector_store, mock_langfuse_objects):
        trace, generation = mock_langfuse_objects
        assistant, client_cls = build_assistant(mock_vector_store, provider="runpod")

        mock_client = client_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_completion("OPT answer"))

        result = await assistant.answer_question("What is OPT?", trace=trace, generation=generation)

        assert result["answer"] == "OPT answer"
        assert len(result["sources"]) == 3
        assert result["metadata"]["model"] == "test-runpod-model"
        assert "response_time" in result["metadata"]
        mock_vector_store.similarity_search.assert_called_once_with("What is OPT?", k=3)

        # generation.end must be called exactly once per request
        end_calls = [c for c in generation.calls]
        assert len(end_calls) == 1

    @pytest.mark.asyncio
    async def test_answer_question_falls_back_to_openai(self, mock_vector_store):
        assistant, client_cls = build_assistant(mock_vector_store, provider="both")

        runpod_client, openai_client = client_cls.call_args_list[0], client_cls.call_args_list[1]

        primary = MagicMock()
        primary.chat.completions.create = AsyncMock(side_effect=RuntimeError("runpod down"))
        fallback = MagicMock()
        fallback.chat.completions.create = AsyncMock(return_value=_completion("openai answer"))

        assistant.client = primary
        assistant.fallback_client = fallback

        result = await assistant.answer_question("question")

        assert result["answer"] == "openai answer"
        fallback.chat.completions.create.assert_awaited_once()
        assert fallback.chat.completions.create.await_args.kwargs["model"] == "test-openai-model"

    @pytest.mark.asyncio
    async def test_answer_question_no_provider_raises(self, mock_vector_store):
        assistant, _ = build_assistant(mock_vector_store, provider=None)

        with pytest.raises(RuntimeError, match="No LLM provider configured"):
            await assistant.answer_question("question")

    @pytest.mark.asyncio
    async def test_answer_question_all_providers_fail(self, mock_vector_store, mock_langfuse_objects):
        trace, generation = mock_langfuse_objects
        assistant, _ = build_assistant(mock_vector_store, provider="both")

        primary = MagicMock()
        primary.chat.completions.create = AsyncMock(side_effect=RuntimeError("fail-1"))
        fallback = MagicMock()
        fallback.chat.completions.create = AsyncMock(side_effect=RuntimeError("fail-2"))
        assistant.client = primary
        assistant.fallback_client = fallback

        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            await assistant.answer_question("q", trace=trace, generation=generation)

        # Error path reported to observability
        assert any(c.get("level") == "ERROR" for c in generation.calls)


class TestDocuments:
    """Tests for document management."""

    def test_list_documents_groups_by_source(self, mock_vector_store):
        from langchain_core.documents import Document

        store = MagicMock()
        store.index.ntotal = 4
        store.docstore._dict = {
            "1": Document(page_content="c1", metadata={"source": "a.pdf"}),  # no type
            "2": Document(page_content="c2", metadata={"source": "a.pdf", "document_type": "immigration"}),
            "3": Document(page_content="c3", metadata={"source": "b.pdf"}),
            "4": Document(page_content="c4"),  # no source at all
        }

        assistant, _ = build_assistant(mock_vector_store)
        assistant.vector_store = store

        result = assistant.list_documents()

        assert result["status"] == "success"
        assert result["total_chunks"] == 4
        assert result["document_count"] == 2
        by_source = {d["source"]: d for d in result["documents"]}
        assert by_source["a.pdf"]["chunk_count"] == 2
        # Known type on a later chunk wins over unknown
        assert by_source["a.pdf"]["document_type"] == "immigration"
        assert by_source["b.pdf"]["document_type"] == "unknown"

    def test_remove_documents_not_implemented(self, mock_vector_store):
        assistant, _ = build_assistant(mock_vector_store)
        result = assistant.remove_documents(["id1"])
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_add_documents_merges_and_returns_counts(self, mock_vector_store):
        from unittest.mock import patch as p
        from langchain_core.documents import Document

        merged_store = MagicMock()
        merged_store.index.ntotal = 5

        assistant, _ = build_assistant(mock_vector_store)

        chunks = [
            Document(page_content="c1", metadata={"source": "doc.pdf"}),
            Document(page_content="c2", metadata={"source": "doc.pdf"}),
        ]

        with p("src.llm.assistant.process_documents", new_callable=AsyncMock) as mock_proc, \
             p("src.llm.assistant.append_to_vector_store", return_value=merged_store) as mock_append:
            mock_proc.return_value = {"chunks": chunks, "documents": ["d"]}

            result = await assistant.add_documents("/tmp/doc.pdf", document_type="immigration")

        assert result["status"] == "success"
        assert result["document_count"] == 1
        assert result["chunk_count"] == 2
        assert result["total_vectors"] == 5

        # document_type must be propagated onto every chunk for citation/filtering
        assert all(c.metadata["document_type"] == "immigration" for c in chunks)
        mock_append.assert_called_once()
        assert assistant.vector_store is merged_store

    @pytest.mark.asyncio
    async def test_add_documents_no_chunks(self, mock_vector_store):
        from unittest.mock import patch as p

        assistant, _ = build_assistant(mock_vector_store)

        with p("src.llm.assistant.process_documents", new_callable=AsyncMock) as mock_proc:
            mock_proc.return_value = {"chunks": []}

            result = await assistant.add_documents("/tmp/empty.pdf")

        assert result["status"] == "error"
        assert "No chunks" in result["message"]

    @pytest.mark.asyncio
    async def test_add_documents_handles_exception(self, mock_vector_store):
        from unittest.mock import patch as p

        assistant, _ = build_assistant(mock_vector_store)

        with p("src.llm.assistant.process_documents", side_effect=RuntimeError("disk full")):
            result = await assistant.add_documents("/tmp/doc.pdf")

        assert result["status"] == "error"
        assert "disk full" in result["message"]


class TestProviderChain:
    """Tests for provider ordering."""

    def test_runpod_first_then_openai(self, mock_vector_store):
        assistant, _ = build_assistant(mock_vector_store, provider="both")
        chain = assistant._provider_chain()
        assert [m for _, m in chain] == ["test-runpod-model", "test-openai-model"]

    def test_only_openai(self, mock_vector_store):
        assistant, _ = build_assistant(mock_vector_store, provider="openai")
        chain = assistant._provider_chain()
        assert [m for _, m in chain] == ["test-openai-model"]


class TestStreamResponse:
    """Tests for the streaming RAG flow."""

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, mock_vector_store):
        assistant, client_cls = build_assistant(mock_vector_store, provider="runpod")

        mock_client = client_cls.return_value
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_iter(make_stream_chunks(["Hello ", "world"]))
        )

        tokens = []
        async for token in assistant.astream_response("q"):
            tokens.append(token)

        assert tokens == ["Hello ", "world"]

    @pytest.mark.asyncio
    async def test_stream_handles_empty_choices_chunk(self, mock_vector_store):
        """Final chunks with empty choice lists must not raise IndexError."""
        assistant, client_cls = build_assistant(mock_vector_store, provider="runpod")

        mock_client = client_cls.return_value
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_iter(make_stream_chunks(["token"]))
        )

        tokens = [t async for t in assistant.astream_response("q")]
        assert tokens == ["token"]

    @pytest.mark.asyncio
    async def test_stream_respects_cancel_event(self, mock_vector_store):
        from threading import Event

        assistant, client_cls = build_assistant(mock_vector_store, provider="runpod")

        cancel = Event()

        async def never_ending_stream():
            # Would keep yielding forever without cancellation
            while True:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="x"))])
                await asyncio.sleep(0.01)

        mock_client = client_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=never_ending_stream())

        received = []

        async def consume():
            async for token in assistant.astream_response("q", cancel_event=cancel):
                received.append(token)
                if len(received) >= 2:
                    cancel.set()

        await asyncio.wait_for(consume(), timeout=5)
        assert len(received) == 2  # stopped right after cancellation

    @pytest.mark.asyncio
    async def test_stream_cancelled_before_start(self, mock_vector_store):
        from threading import Event

        assistant, _ = build_assistant(mock_vector_store, provider="runpod")
        cancel = Event()
        cancel.set()

        tokens = [t async for t in assistant.astream_response("q", cancel_event=cancel)]
        assert tokens == []

    @pytest.mark.asyncio
    async def test_stream_error_yields_error_message(self, mock_vector_store, mock_langfuse_objects):
        trace, generation = mock_langfuse_objects
        assistant, client_cls = build_assistant(mock_vector_store, provider="runpod")

        mock_client = client_cls.return_value
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

        tokens = [t async for t in assistant.astream_response("q", trace=trace, generation=generation)]
        assert len(tokens) == 1
        assert tokens[0].startswith("Error:")
