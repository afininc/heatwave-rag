# HeatWave RAG

A comprehensive RAG (Retrieval-Augmented Generation) library for Oracle HeatWave MySQL with LangChain integration.

## Features

- =€ **HeatWave Vector Store**: Optimized for Oracle HeatWave MySQL's vector capabilities
- = **LangChain Integration**: Seamless integration with LangChain ecosystem
- =Ä **Multi-Format Support**: Process PDF, DOCX, Markdown, JSON, CSV, and text files
- = **Hybrid Search**: Combine vector similarity and keyword search
- <â **Multi-Tenancy**: Built-in support for projects and namespaces
- = **Flexible Embeddings**: Support for OpenAI, HuggingFace, Cohere, and more
- <¯ **Production Ready**: Connection pooling, batch processing, and error handling

## Installation

```bash
pip install heatwave-rag
```

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
```

### Connection Configuration

```python
from heatwave_rag import HeatWaveRAG, ConnectionConfig

config = ConnectionConfig(
    host="localhost",
    port=3306,
    user="root",
    password="password",
    database="mydb",
    pool_size=5,
    max_overflow=10
)

rag = HeatWaveRAG(connection_config=config)
```

## Advanced Usage

### Custom Embeddings

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

rag.set_embeddings(embeddings)
```

### Document Processing with Custom Chunking

```python
# Add documents with custom chunk settings
doc_ids = rag.add_documents(
    documents=["large_document.pdf"],
    chunk_size=500,
    chunk_overlap=50,
    metadata={
        "source": "technical_manual",
        "version": "2.0",
        "tags": ["manual", "technical"]
    }
)
```

### Hybrid Search

```python
from heatwave_rag import VectorSearchQuery

# Create custom search query
query = VectorSearchQuery(
    query_text="machine learning algorithms",
    top_k=10,
    filters={"tags": ["AI", "ML"]},
    project="research",
    namespace="papers"
)

results = rag.search(query, similarity_threshold=0.7)
```

### Batch Processing

```python
# Process entire directory
doc_ids = rag.add_directory(
    directory_path="./documents",
    glob_pattern="**/*.pdf",
    recursive=True,
    exclude=["**/drafts/*"],
    batch_size=50
)
```

### Custom RAG Prompts

```python
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
| vector | VECTOR | Embedding vector |
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

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and feature requests, please use the [GitHub issue tracker](https://github.com/afininc/heatwave-rag/issues).