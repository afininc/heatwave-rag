"""Pydantic schemas for data validation."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    """Metadata for a document."""

    source: Optional[str] = Field(None, max_length=500)
    source_id: Optional[str] = Field(None, max_length=255)
    document_title: Optional[str] = Field(None, max_length=500)
    author: Optional[str] = Field(None, max_length=255)
    tags: Optional[list[str]] = None
    lang: str = Field("en", max_length=10)
    project: Optional[str] = Field(None, max_length=255)
    namespace: Optional[str] = Field(None, max_length=255)
    custom_metadata: Optional[dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class DocumentChunk(BaseModel):
    """A chunk of document with text and metadata."""

    text: str
    meta: dict[str, Any] = Field(default_factory=dict)
    chunk_index: int = 0


class VectorSearchQuery(BaseModel):
    """Query parameters for vector search."""

    query_text: str
    top_k: int = Field(10, ge=1, le=100)
    project: Optional[str] = None
    namespace: Optional[str] = None
    filters: Optional[dict[str, Any]] = None
    include_metadata: bool = True


class VectorSearchResult(BaseModel):
    """Result from vector search."""

    id: int
    text: str
    score: float
    meta: Optional[dict[str, Any]] = None
    source: Optional[str] = None
    chunk_index: int


class TableInitConfig(BaseModel):
    """Configuration for table initialization."""

    table_name: str = "vector_documents"
    vector_dimension: int = Field(1536, ge=1)  # Default for OpenAI ada-002
    drop_if_exists: bool = False
    create_indexes: bool = True


class ConnectionConfig(BaseModel):
    """Database connection configuration."""

    host: str
    port: int = 3306
    user: str
    password: str
    database: str
    charset: str = "utf8mb4"
    pool_size: int = Field(5, ge=1, le=100)
    max_overflow: int = Field(10, ge=0, le=100)
    pool_timeout: int = Field(30, ge=1)
    pool_recycle: int = Field(3600, ge=60)
