"""HeatWave RAG - A comprehensive RAG library for Oracle HeatWave MySQL with LangChain integration."""

__version__ = "0.1.0"
__all__ = [
    "HeatWaveRAG",
    "DocumentChunk",
    "DocumentMetadata",
    "VectorSearchQuery",
    "VectorSearchResult",
]

from heatwave_rag.core import HeatWaveRAG
from heatwave_rag.schemas import (
    DocumentChunk,
    DocumentMetadata,
    VectorSearchQuery,
    VectorSearchResult,
)
