# Testing OPT-RAG Assistant

This directory contains tests for the OPT-RAG Assistant, which forms the core of the RAG pipeline for the International Student Visa Assistant.

## Test Structure

- **Unit Tests** (`test_assistant.py`): Testing individual components with mocked dependencies
- **Integration Tests** (`test_integration.py`): Testing interactions between components with minimal mocking
- **Manual Testing** (`manual_test_assistant.py` in the project root): Script for interactive testing with real models and documents

## Testing Approaches

The testing approach follows a pyramid structure:

1. **Automated Tests** (in this directory)
   - Unit tests with mocked dependencies for fast, reliable testing
   - Integration tests to verify component interactions
   - All tests run without requiring actual LLM models

2. **Manual Interactive Testing** (using `manual_test_assistant.py` in project root)
   - For testing with real models and documents
   - Interactive CLI-based testing
   - Performs actual LLM inference

## Running Tests

### Unit Tests

Run the basic unit tests (these mock all external dependencies):

```bash
# From project root
pytest rag-pipeline/tests/test_llm/test_assistant.py -v
```

### Integration Tests

Integration tests require setting an environment variable to run:

```bash
# From project root
RUN_INTEGRATION_TESTS=1 pytest rag-pipeline/tests/test_llm/test_integration.py -v
```

> **Note**: Integration tests will still mock the LLM to avoid requiring a full model download, but they use real document processing and vector stores.

### Manual Testing

For comprehensive testing with a real model, use the `manual_test_assistant.py` script:

```bash
# From project root
python rag-pipeline/manual_test_assistant.py --model models/Qwen2.5-0.5B-Instruct --document examples/visa_faq.pdf --query "What is OPT?"
```

Additional options:
- `--vector-store`: Path to vector store directory (default: `vector_store`)
- `--device`: Specify device (cuda, mps, cpu)
- `--stream`: Stream the response instead of waiting for completion

## Test Data

The `examples` directory contains sample documents for testing purposes:

- `sample_visa_info.txt`: Basic OPT visa information

## Test Coverage

The tests cover the following aspects of the OPT-RAG Assistant:

1. **Initialization**:
   - Hardware detection
   - Model loading for different devices
   - Vector store initialization

2. **Document Processing**:
   - Adding documents to the vector store
   - Retrieving document metadata

3. **Querying**:
   - Standard question answering
   - Streaming responses
   - Error handling

4. **End-to-End Flow**:
   - Document ingestion to query response 