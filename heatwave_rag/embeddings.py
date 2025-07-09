"""Embedding generation and vector storage utilities."""

import logging
from typing import Optional, Union

import numpy as np
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import (
    CohereEmbeddings,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
)

from heatwave_rag.database import DatabaseManager
from heatwave_rag.models import VectorDocument
from heatwave_rag.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding generation and storage."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        embeddings: Optional[Embeddings] = None,
        embedding_model_name: Optional[str] = None,
    ):
        """Initialize embedding manager.

        Args:
            db_manager: Database manager instance
            embeddings: LangChain embeddings instance
            embedding_model_name: Name of the embedding model
        """
        self.db_manager = db_manager
        self.embeddings = embeddings
        self.embedding_model_name = embedding_model_name or "unknown"

        if self.embeddings and not self.embedding_model_name:
            # Try to infer model name from embeddings instance
            self._infer_model_name()

    def _infer_model_name(self) -> None:
        """Infer embedding model name from embeddings instance."""
        if isinstance(self.embeddings, OpenAIEmbeddings):
            self.embedding_model_name = getattr(
                self.embeddings, "model", "text-embedding-ada-002"
            )
        elif isinstance(self.embeddings, HuggingFaceEmbeddings):
            self.embedding_model_name = getattr(
                self.embeddings, "model_name", "sentence-transformers/all-MiniLM-L6-v2"
            )
        elif isinstance(self.embeddings, CohereEmbeddings):
            self.embedding_model_name = "cohere-embed"
        else:
            self.embedding_model_name = type(self.embeddings).__name__

    def set_embeddings(
        self, embeddings: Embeddings, model_name: Optional[str] = None
    ) -> None:
        """Set the embeddings instance.

        Args:
            embeddings: LangChain embeddings instance
            model_name: Optional model name override
        """
        self.embeddings = embeddings
        self.embedding_model_name = model_name
        if not model_name:
            self._infer_model_name()

    def embed_text(
        self, text: Union[str, list[str]]
    ) -> Union[list[float], list[list[float]]]:
        """Generate embeddings for text.

        Args:
            text: Single text or list of texts

        Returns:
            Embedding vector(s)
        """
        if not self.embeddings:
            raise ValueError("No embeddings instance configured")

        if isinstance(text, str):
            return self.embeddings.embed_query(text)
        else:
            return self.embeddings.embed_documents(text)

    def store_documents(
        self,
        chunks: list[DocumentChunk],
        batch_size: int = 100,
        skip_existing: bool = True,
    ) -> list[int]:
        """Store document chunks with embeddings in the database.

        Args:
            chunks: List of document chunks
            batch_size: Batch size for processing
            skip_existing: Skip chunks that already exist in DB

        Returns:
            List of stored document IDs
        """
        if not self.embeddings:
            raise ValueError("No embeddings instance configured")

        stored_ids = []

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Generate embeddings for batch
            texts = [chunk.text for chunk in batch]
            embeddings = self.embed_text(texts)

            with self.db_manager.get_session() as session:
                for chunk, embedding in zip(batch, embeddings):
                    # Check if document already exists
                    if skip_existing and chunk.meta.get("source_id"):
                        existing = (
                            session.query(VectorDocument)
                            .filter_by(
                                source_id=chunk.meta.get("source_id"),
                                chunk_index=chunk.chunk_index,
                            )
                            .first()
                        )

                        if existing:
                            logger.info(
                                f"Skipping existing chunk: {chunk.meta.get('source_id')}:{chunk.chunk_index}"
                            )
                            continue

                    # Create new document
                    doc = self._create_document(chunk, embedding)
                    session.add(doc)
                    session.flush()
                    stored_ids.append(doc.id)

            logger.info(
                f"Stored {len(stored_ids)} documents in batch {i//batch_size + 1}"
            )

        return stored_ids

    def _create_document(
        self, chunk: DocumentChunk, embedding: list[float]
    ) -> VectorDocument:
        """Create a VectorDocument instance from chunk and embedding.

        Args:
            chunk: Document chunk
            embedding: Embedding vector

        Returns:
            VectorDocument instance
        """
        metadata = chunk.meta

        # Prepare metadata JSON
        metadata_json = {
            **metadata,
            "chunk_metadata": {
                "chunk_index": chunk.chunk_index,
                "embedding_model": self.embedding_model_name,
            },
        }

        return VectorDocument(
            vector=np.array(embedding),
            text=chunk.text,
            meta=metadata_json,
            source=metadata.get("source"),
            source_id=metadata.get("source_id"),
            chunk_index=chunk.chunk_index,
            document_title=metadata.get("document_title"),
            author=metadata.get("author"),
            tags=metadata.get("tags"),
            lang=metadata.get("lang", "en"),
            project=metadata.get("project"),
            namespace=metadata.get("namespace"),
            embedding_model=self.embedding_model_name,
        )

    def update_embeddings(
        self,
        document_ids: list[int],
        new_embeddings: Optional[Embeddings] = None,
        new_model_name: Optional[str] = None,
    ) -> int:
        """Update embeddings for existing documents.

        Args:
            document_ids: List of document IDs to update
            new_embeddings: New embeddings instance (uses current if None)
            new_model_name: New model name

        Returns:
            Number of documents updated
        """
        if new_embeddings:
            old_embeddings = self.embeddings
            old_model_name = self.embedding_model_name
            self.set_embeddings(new_embeddings, new_model_name)

        updated_count = 0

        try:
            with self.db_manager.get_session() as session:
                # Fetch documents
                documents = (
                    session.query(VectorDocument)
                    .filter(VectorDocument.id.in_(document_ids))
                    .all()
                )

                # Generate new embeddings
                texts = [doc.text for doc in documents]
                embeddings = self.embed_text(texts)

                # Update documents
                for doc, embedding in zip(documents, embeddings):
                    doc.vector = np.array(embedding)
                    doc.embedding_model = self.embedding_model_name
                    updated_count += 1

                session.commit()
                logger.info(f"Updated embeddings for {updated_count} documents")

        finally:
            # Restore original embeddings if changed
            if new_embeddings:
                self.embeddings = old_embeddings
                self.embedding_model_name = old_model_name

        return updated_count

    def delete_documents(
        self,
        document_ids: Optional[list[int]] = None,
        source_id: Optional[str] = None,
        project: Optional[str] = None,
        namespace: Optional[str] = None,
        hard_delete: bool = False,
    ) -> int:
        """Delete documents from the database.

        Args:
            document_ids: List of document IDs
            source_id: Delete all documents with this source ID
            project: Delete all documents in this project
            namespace: Delete all documents in this namespace
            hard_delete: Permanently delete instead of soft delete

        Returns:
            Number of documents deleted
        """
        with self.db_manager.get_session() as session:
            query = session.query(VectorDocument)

            # Build filter conditions
            if document_ids:
                query = query.filter(VectorDocument.id.in_(document_ids))
            if source_id:
                query = query.filter(VectorDocument.source_id == source_id)
            if project:
                query = query.filter(VectorDocument.project == project)
            if namespace:
                query = query.filter(VectorDocument.namespace == namespace)

            if hard_delete:
                count = query.delete()
            else:
                count = query.update({"is_deleted": True})

            session.commit()
            logger.info(
                f"{'Hard' if hard_delete else 'Soft'} deleted {count} documents"
            )

            return count

    def get_document_count(
        self,
        project: Optional[str] = None,
        namespace: Optional[str] = None,
        include_deleted: bool = False,
    ) -> int:
        """Get count of documents in the database.

        Args:
            project: Filter by project
            namespace: Filter by namespace
            include_deleted: Include soft-deleted documents

        Returns:
            Document count
        """
        with self.db_manager.get_session() as session:
            query = session.query(VectorDocument)

            if project:
                query = query.filter(VectorDocument.project == project)
            if namespace:
                query = query.filter(VectorDocument.namespace == namespace)
            if not include_deleted:
                query = query.filter(~VectorDocument.is_deleted)

            return query.count()
