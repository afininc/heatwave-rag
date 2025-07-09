"""Core HeatWave RAG API."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings

from heatwave_rag.database import DatabaseManager
from heatwave_rag.document_processor import DocumentProcessor
from heatwave_rag.embeddings import EmbeddingManager
from heatwave_rag.rag import RAGEngine
from heatwave_rag.schemas import (
    ConnectionConfig,
    DocumentChunk,
    TableInitConfig,
    VectorSearchQuery,
    VectorSearchResult,
)
from heatwave_rag.vector_search import VectorSearchEngine

logger = logging.getLogger(__name__)


class HeatWaveRAG:
    """Main interface for HeatWave RAG operations."""

    def __init__(
        self,
        connection_config: Union[ConnectionConfig, dict[str, Any]],
        embeddings: Optional[Embeddings] = None,
        llm: Optional[BaseChatModel] = None,
        table_name: str = "vector_documents",
        vector_dimension: int = 1536,
        auto_init: bool = True,
    ):
        """Initialize HeatWave RAG.

        Args:
            connection_config: Database connection configuration
            embeddings: LangChain embeddings instance
            llm: LangChain LLM instance for RAG
            table_name: Name of the vector table
            vector_dimension: Dimension of embedding vectors
            auto_init: Automatically initialize tables
        """
        # Convert dict to ConnectionConfig if needed
        if isinstance(connection_config, dict):
            connection_config = ConnectionConfig(**connection_config)

        self.connection_config = connection_config
        self.table_name = table_name
        self.vector_dimension = vector_dimension

        # Initialize components
        self.db_manager = DatabaseManager(connection_config)
        self.embedding_manager = EmbeddingManager(self.db_manager, embeddings)
        self.vector_search_engine = VectorSearchEngine(
            self.db_manager, self.embedding_manager
        )
        self.document_processor = DocumentProcessor()
        self.rag_engine = RAGEngine(self.vector_search_engine, llm)

        # Initialize tables if requested
        if auto_init:
            self.init_tables()

        logger.info("HeatWave RAG initialized successfully")

    def init_tables(
        self,
        drop_if_exists: bool = False,
        create_indexes: bool = True,
    ) -> None:
        """Initialize database tables.

        Args:
            drop_if_exists: Drop existing tables before creating
            create_indexes: Create performance indexes
        """
        table_config = TableInitConfig(
            table_name=self.table_name,
            vector_dimension=self.vector_dimension,
            drop_if_exists=drop_if_exists,
            create_indexes=create_indexes,
        )

        self.db_manager.init_tables(table_config)
        logger.info(f"Initialized table: {self.table_name}")

    def verify_table(self) -> dict[str, Any]:
        """Verify if existing table is compatible.

        Returns:
            Compatibility check results
        """
        return self.db_manager.verify_table_compatibility(self.table_name)

    def set_embeddings(
        self, embeddings: Embeddings, model_name: Optional[str] = None
    ) -> None:
        """Set embeddings instance.

        Args:
            embeddings: LangChain embeddings instance
            model_name: Optional model name override
        """
        self.embedding_manager.set_embeddings(embeddings, model_name)

    def set_llm(self, llm: BaseChatModel) -> None:
        """Set LLM instance for RAG.

        Args:
            llm: LangChain LLM instance
        """
        self.rag_engine.set_llm(llm)

    def add_documents(
        self,
        documents: Union[str, Path, list[str], list[Path], list[DocumentChunk]],
        metadata: Optional[dict[str, Any]] = None,
        batch_size: int = 100,
        skip_existing: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[int]:
        """Add documents to the vector store.

        Args:
            documents: Documents to add (file paths, text, or chunks)
            metadata: Metadata to attach to documents
            batch_size: Batch size for processing
            skip_existing: Skip documents that already exist
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks

        Returns:
            List of stored document IDs
        """
        chunks = []

        # Process different input types
        if isinstance(documents, (str, Path)):
            documents = [documents]

        for doc in documents:
            if isinstance(doc, DocumentChunk):
                chunks.append(doc)
            elif isinstance(doc, (str, Path)):
                doc_path = Path(doc)
                if doc_path.exists() and doc_path.is_file():
                    # Process file
                    doc_chunks = self.document_processor.process_file(
                        str(doc_path),
                        metadata,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    chunks.extend(doc_chunks)
                else:
                    # Process as text
                    text_chunks = self.document_processor.process_text(
                        str(doc),
                        metadata,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    chunks.extend(text_chunks)

        # Store chunks with embeddings
        doc_ids = self.embedding_manager.store_documents(
            chunks, batch_size, skip_existing
        )

        logger.info(f"Added {len(doc_ids)} documents to vector store")
        return doc_ids

    def add_directory(
        self,
        directory_path: Union[str, Path],
        glob_pattern: str = "**/*",
        recursive: bool = True,
        exclude: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        batch_size: int = 100,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[int]:
        """Add all documents from a directory.

        Args:
            directory_path: Path to directory
            glob_pattern: Glob pattern for file matching
            recursive: Search recursively
            exclude: Patterns to exclude
            metadata: Metadata to attach
            batch_size: Batch size for processing
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks

        Returns:
            List of stored document IDs
        """
        # Load documents from directory
        documents = self.document_processor.load_directory(
            str(directory_path),
            glob_pattern,
            recursive,
            exclude,
        )

        # Process and split documents
        chunks = []
        for doc in documents:
            # Merge metadata
            merged_metadata = doc.metadata.copy()
            if metadata:
                merged_metadata.update(metadata)

            # Split into chunks
            doc_chunks = self.document_processor.process_text(
                doc.page_content,
                merged_metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks.extend(doc_chunks)

        # Store chunks
        doc_ids = self.embedding_manager.store_documents(chunks, batch_size)

        logger.info(f"Added {len(doc_ids)} documents from directory: {directory_path}")
        return doc_ids

    def search(
        self,
        query: Union[str, VectorSearchQuery],
        top_k: int = 10,
        project: Optional[str] = None,
        namespace: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: float = 0.0,
        include_metadata: bool = True,
    ) -> list[VectorSearchResult]:
        """Search for similar documents.

        Args:
            query: Search query (text or VectorSearchQuery)
            top_k: Number of results to return
            project: Filter by project
            namespace: Filter by namespace
            filters: Additional filters
            similarity_threshold: Minimum similarity score
            include_metadata: Include metadata in results

        Returns:
            List of search results
        """
        if isinstance(query, str):
            query = VectorSearchQuery(
                query_text=query,
                top_k=top_k,
                project=project,
                namespace=namespace,
                filters=filters,
                include_metadata=include_metadata,
            )

        return self.vector_search_engine.search(query, similarity_threshold)

    def query(
        self,
        query: str,
        top_k: int = 5,
        project: Optional[str] = None,
        namespace: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        return_sources: bool = True,
        prompt_template: Optional[str] = None,
        **llm_kwargs,
    ) -> dict[str, Any]:
        """Perform RAG query.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            project: Filter by project
            namespace: Filter by namespace
            filters: Additional filters
            return_sources: Include source documents
            prompt_template: Custom prompt template
            **llm_kwargs: Additional LLM arguments

        Returns:
            Query response with answer and sources
        """
        return self.rag_engine.query(
            query=query,
            top_k=top_k,
            project=project,
            namespace=namespace,
            filters=filters,
            return_sources=return_sources,
            prompt_template=prompt_template,
            **llm_kwargs,
        )

    def delete_documents(
        self,
        document_ids: Optional[list[int]] = None,
        source_id: Optional[str] = None,
        project: Optional[str] = None,
        namespace: Optional[str] = None,
        hard_delete: bool = False,
    ) -> int:
        """Delete documents from vector store.

        Args:
            document_ids: IDs to delete
            source_id: Delete by source ID
            project: Delete by project
            namespace: Delete by namespace
            hard_delete: Permanently delete

        Returns:
            Number of deleted documents
        """
        return self.embedding_manager.delete_documents(
            document_ids, source_id, project, namespace, hard_delete
        )

    def update_embeddings(
        self,
        document_ids: list[int],
        new_embeddings: Optional[Embeddings] = None,
        new_model_name: Optional[str] = None,
    ) -> int:
        """Update embeddings for existing documents.

        Args:
            document_ids: Document IDs to update
            new_embeddings: New embeddings instance
            new_model_name: New model name

        Returns:
            Number of updated documents
        """
        return self.embedding_manager.update_embeddings(
            document_ids, new_embeddings, new_model_name
        )

    def get_document_count(
        self,
        project: Optional[str] = None,
        namespace: Optional[str] = None,
        include_deleted: bool = False,
    ) -> int:
        """Get document count.

        Args:
            project: Filter by project
            namespace: Filter by namespace
            include_deleted: Include soft-deleted documents

        Returns:
            Document count
        """
        return self.embedding_manager.get_document_count(
            project, namespace, include_deleted
        )

    def close(self) -> None:
        """Close database connections."""
        self.db_manager.close()
        logger.info("HeatWave RAG closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
