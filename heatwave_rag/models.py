"""Database models for HeatWave RAG."""

from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Index, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

from heatwave_rag.types import VECTOR

Base = declarative_base()


class VectorDocument(Base):
    """Vector document storage model for HeatWave."""

    __tablename__ = "vector_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vector = Column(VECTOR, nullable=False)  # Vector column for embeddings
    text = Column(Text, nullable=False)  # Original text content
    metadata = Column(JSON, nullable=True)  # Additional metadata as JSON
    source = Column(String(500), nullable=True)  # Source file/URL
    source_id = Column(String(255), nullable=True)  # Unique identifier from source
    chunk_index = Column(
        Integer, nullable=False, default=0
    )  # Index of chunk within source
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    document_title = Column(String(500), nullable=True)  # Document title
    author = Column(String(255), nullable=True)  # Document author
    tags = Column(JSON, nullable=True)  # Tags as JSON array
    lang = Column(String(10), nullable=True, default="en")  # Language code
    project = Column(String(255), nullable=True)  # Project identifier
    namespace = Column(String(255), nullable=True)  # Namespace for multi-tenancy
    is_deleted = Column(Boolean, nullable=False, default=False)  # Soft delete flag
    embedding_model = Column(String(255), nullable=False)  # Model used for embedding

    # Create indexes for better query performance
    __table_args__ = (
        Index("idx_source_id", "source_id"),
        Index("idx_project_namespace", "project", "namespace"),
        Index("idx_created_at", "created_at"),
        Index("idx_is_deleted", "is_deleted"),
    )
