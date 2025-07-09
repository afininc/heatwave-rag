# HeatWave RAG

A comprehensive RAG (Retrieval-Augmented Generation) library for Oracle HeatWave MySQL with LangChain integration.

## Features

- **HeatWave Vector Store**: Optimized for Oracle HeatWave MySQL's vector capabilities
- **LangChain Integration**: Seamless integration with LangChain ecosystem
- **Multi-Format Support**: Process PDF, DOCX, Markdown, JSON, CSV, and text files
- **Hybrid Search**: Combine vector similarity and keyword search
- **Multi-Tenancy**: Built-in support for projects and namespaces
- **Flexible Embeddings**: Support for OpenAI, HuggingFace, Cohere, and more
- **Production Ready**: Connection pooling, batch processing, and error handling

## Installation

### Install from GitHub

Since the package is not yet available on PyPI, install directly from GitHub:

```bash
# Using pip
pip install git+https://github.com/afininc/heatwave-rag.git

# Using uv (recommended)
uv pip install git+https://github.com/afininc/heatwave-rag.git

# For development
git clone https://github.com/afininc/heatwave-rag.git
cd heatwave-rag
uv pip install -e .
```

### Requirements

- Python 3.9+
- MySQL 8.0+ (MySQL 9.0+ recommended for native vector support)
- LangChain and your preferred embedding/LLM providers

## Quick Start

```python
import heatwave_rag
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Initialize HeatWave RAG
rag = heatwave_rag.HeatWaveRAG(
    connection_config={
        "host": "your-mysql-host",
        "port": 3306,
        "user": "your-username",
        "password": "your-password",
        "database": "your-database"
    },
    embeddings=OpenAIEmbeddings(),
    llm=ChatOpenAI(model="gpt-3.5-turbo")
)

# Add documents
doc_ids = rag.add_documents(
    ["path/to/document.pdf", "path/to/document.txt"],
    metadata={"project": "my_project", "namespace": "default"}
)

# Search for similar documents
results = rag.search("What is artificial intelligence?", top_k=5)

# Perform RAG query
response = rag.query(
    "Explain the key concepts of machine learning",
    top_k=3,
    return_sources=True
)

print(response["answer"])
print(response["sources"])
```

## Configuration

### Vector Distance Metrics

HeatWave RAG supports multiple distance metrics for vector similarity search:

- **COSINE** (default): Measures the cosine of the angle between vectors. Best for normalized embeddings.
- **DOT**: Dot product similarity. Suitable when vector magnitude matters.
- **EUCLIDEAN**: Euclidean distance converted to similarity. Good for spatial similarity.

```python
# Use a different distance metric
rag = heatwave_rag.HeatWaveRAG(
    connection_config={...},
    distance_metric="EUCLIDEAN"  # or "DOT", "COSINE"
)
```

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# MySQL Configuration
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=your_username
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=heatwave_rag

# OpenAI (for embeddings and LLM)
OPENAI_API_KEY=your_api_key

# Optional: Other providers
ANTHROPIC_API_KEY=your_api_key
COHERE_API_KEY=your_api_key
HUGGINGFACE_API_KEY=your_api_key
```

## API Reference

### HeatWaveRAG

The main interface for all RAG operations.

```python
from heatwave_rag import HeatWaveRAG

rag = HeatWaveRAG(
    connection_config={...},
    embeddings=...,  # LangChain embeddings
    llm=...,         # LangChain LLM
    table_name="vector_documents",
    vector_dimension=1536,
    auto_init=True
)
```

### Adding Documents

```python
# Add files
doc_ids = rag.add_documents(
    ["file1.pdf", "file2.txt"],
    metadata={"author": "John Doe", "project": "research"}
)

# Add directory
doc_ids = rag.add_directory(
    "path/to/documents",
    glob_pattern="*.pdf",
    metadata={"project": "research"}
)

# Add text directly
doc_ids = rag.add_documents(
    ["Text content 1", "Text content 2"],
    metadata={"source": "manual"}
)
```

### Searching

```python
# Simple search
results = rag.search("quantum computing", top_k=10)

# Advanced search with filters
results = rag.search(
    query="machine learning",
    top_k=5,
    project="research",
    namespace="papers"
)

# Hybrid search (vector + keyword)
results = rag.vector_search_engine.hybrid_search(
    VectorSearchQuery(
        query_text="neural networks",
        top_k=10,
        project="research"
    ),
    keyword_weight=0.3,
    vector_weight=0.7
)
```

### RAG Queries

```python
# Basic RAG query
response = rag.query("What is deep learning?")

# With custom parameters
response = rag.query(
    "Explain transformers",
    top_k=5,
    temperature=0.7,
    return_sources=True,
    project="ml_papers"
)

# Custom prompt template
custom_prompt = """
You are a technical assistant. Use the following context to answer the question.
Be concise and specific.

Context: {context}
Question: {question}
Answer:
"""

response = rag.query(
    "What are the main components?",
    prompt_template=custom_prompt,
    temperature=0.3
)
```

## Database Schema

The library creates a `vector_documents` table with the following structure:

| Column | Type | Description |
|--------|------|-------------|
| id | INT | Primary key |
| vector | LONGTEXT | Embedding vector (stored as JSON) |
| text | TEXT | Document content |
| metadata | JSON | Additional metadata |
| source | VARCHAR(500) | Source file/URL |
| source_id | VARCHAR(255) | Unique source identifier |
| chunk_index | INT | Chunk position in document |
| document_title | VARCHAR(500) | Document title |
| author | VARCHAR(255) | Document author |
| tags | JSON | Document tags |
| lang | VARCHAR(10) | Language code |
| project | VARCHAR(255) | Project identifier |
| namespace | VARCHAR(255) | Namespace for multi-tenancy |
| is_deleted | BOOLEAN | Soft delete flag |
| embedding_model | VARCHAR(255) | Model used for embedding |
| created_at | DATETIME | Creation timestamp |

## Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=heatwave_rag
```

## Development

```bash
# Clone repository
git clone https://github.com/afininc/heatwave-rag.git
cd heatwave-rag

# Install with uv
uv sync --dev

# Run linting
ruff check . --fix

# Format code
ruff format .
```

## Utility Scripts

The `scripts/` directory contains useful utilities:

### check_mysql_version.py
Check MySQL version and vector function support:
```bash
python scripts/check_mysql_version.py
```

### debug_vectors.py
Debug vector storage and inspect table structure:
```bash
python scripts/debug_vectors.py
```

### recreate_table.py
Recreate the vector_documents table:
```bash
python scripts/recreate_table.py
```

### test_pdf_demo.py
Test the library with PDF files:
```bash
python scripts/test_pdf_demo.py
```

## MySQL Vector Support

HeatWave RAG works with:
- **MySQL 8.0+**: Uses LONGTEXT column with Python-based vector search
- **MySQL 9.0+**: Can leverage native VECTOR type and DISTANCE() function when available

The library automatically detects available features and uses the most efficient approach.

## License

Apache License 2.0 - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.