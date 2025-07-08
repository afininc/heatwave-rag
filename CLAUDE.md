# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HeatWave RAG is a Python library that integrates Oracle HeatWave MySQL's vector storage capabilities with LangChain for building RAG (Retrieval-Augmented Generation) applications. The library is designed for Oracle Cloud Tier 1 partners to leverage HeatWave's vector store functionality efficiently.

## Development Commands

### Package Management (using uv)
```bash
# Add dependencies
uv add <package-name>

# Remove dependencies
uv remove <package-name>

# Install development dependencies
uv sync --dev
```

### Code Quality
```bash
# Run linting and auto-fix
ruff check . --fix

# Format code
ruff format .

# Run both lint and format
ruff check . --fix && ruff format .
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=heatwave_rag

# Run a specific test file
pytest tests/test_<module>.py

# Run tests matching a pattern
pytest -k "test_pattern"
```

### Git Workflow
```bash
# Commit with Conventional Commits format
git commit -m "<type>(<scope>): <description>"

# Types: feat, fix, docs, style, refactor, perf, test, chore
```

## Architecture

### Core Components

1. **HeatWaveRAG** (`core.py`): Main API interface that orchestrates all components
   - Handles initialization, document management, search, and RAG queries
   - Context manager support for proper resource cleanup

2. **DatabaseManager** (`database.py`): Manages MySQL HeatWave connections
   - Connection pooling with SQLAlchemy
   - Table initialization and schema verification
   - Session management with context managers

3. **VectorDocument** (`models.py`): SQLAlchemy ORM model for vector storage
   - Custom VECTOR type for MySQL compatibility
   - Comprehensive metadata fields (source, tags, language, namespace, etc.)
   - Soft delete support with `is_deleted` flag

4. **EmbeddingManager** (`embeddings.py`): Handles embedding generation and storage
   - Supports multiple embedding models (OpenAI, HuggingFace, Cohere)
   - Batch processing for efficiency
   - Document deduplication via source_id

5. **VectorSearchEngine** (`vector_search.py`): Vector similarity search
   - MySQL native vector search (when available)
   - Python fallback with cosine similarity
   - Hybrid search combining vector and keyword search

6. **DocumentProcessor** (`document_processor.py`): LangChain integration for document processing
   - Multiple file format support (PDF, DOCX, MD, JSON, CSV, TXT)
   - Configurable text splitting strategies
   - Directory processing with glob patterns

7. **RAGEngine** (`rag.py`): RAG query execution
   - Retrieval + generation pipeline
   - Customizable prompts
   - Streaming support for real-time responses

### Data Flow

1. **Document Ingestion**: Files/text → DocumentProcessor → chunks → EmbeddingManager → VectorDocument → MySQL
2. **Search**: Query → EmbeddingManager → vector → VectorSearchEngine → MySQL/Python similarity → results
3. **RAG**: Query → VectorSearchEngine → context → RAGEngine + LLM → answer

### Key Design Decisions

- **Table Flexibility**: Users can specify custom table names and verify existing table compatibility
- **Multi-tenancy**: Built-in support via `project` and `namespace` fields
- **LangChain Integration**: Leverages LangChain's document loaders and text splitters for flexibility
- **Embedding Model Agnostic**: Works with any LangChain-compatible embedding model
- **MySQL HeatWave Optimization**: Attempts native vector operations, falls back to Python when unavailable

## Important Conventions

1. **Language**: All user interactions in Korean, code and comments in English
2. **Error Handling**: Use proper exception handling with meaningful error messages
3. **Type Hints**: Always use type hints for function parameters and returns
4. **Commit Messages**: Follow Conventional Commits format (see COMMIT.md)
5. **Python Version**: Target Python 3.9+ compatibility

## Database Schema Requirements

The vector_documents table must have these columns:
- id, vector, text, metadata, source, source_id, chunk_index, created_at
- document_title, author, tags, lang, project, namespace, is_deleted, embedding_model

## Testing Considerations

When writing tests, you'll need to provide:
- MySQL connection info (host, port, user, password, database)
- LangChain API keys and base URLs for embeddings/LLMs
- Test data in various formats for document processing