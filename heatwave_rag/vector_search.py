"""Vector search functionality for HeatWave."""

import logging
from typing import Any, Optional

import numpy as np
from sqlalchemy import and_, text
from sqlalchemy.orm import Session

from heatwave_rag.database import DatabaseManager
from heatwave_rag.embeddings import EmbeddingManager
from heatwave_rag.models import VectorDocument
from heatwave_rag.schemas import VectorSearchQuery, VectorSearchResult

logger = logging.getLogger(__name__)


class VectorSearchEngine:
    """Performs vector similarity search in HeatWave."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_manager: EmbeddingManager,
    ):
        """Initialize vector search engine.

        Args:
            db_manager: Database manager instance
            embedding_manager: Embedding manager instance
        """
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager

    def search(
        self,
        query: VectorSearchQuery,
        similarity_threshold: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Perform vector similarity search.

        Args:
            query: Search query parameters
            similarity_threshold: Minimum similarity score

        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_text(query.query_text)
        query_vector = np.array(query_embedding)

        with self.db_manager.get_session() as session:
            results = self._execute_search(
                session, query_vector, query, similarity_threshold
            )

            # Convert to VectorSearchResult objects
            search_results = []
            for doc, score in results:
                result = VectorSearchResult(
                    id=doc.id,
                    text=doc.text,
                    score=float(score),
                    source=doc.source,
                    chunk_index=doc.chunk_index,
                )

                if query.include_metadata:
                    result.meta = self._build_metadata(doc)

                search_results.append(result)

            return search_results

    def _execute_search(
        self,
        session: Session,
        query_vector: np.ndarray,
        query: VectorSearchQuery,
        similarity_threshold: float,
    ) -> list[tuple[VectorDocument, float]]:
        """Execute the vector search query.

        Args:
            session: Database session
            query_vector: Query embedding vector
            query: Search parameters
            similarity_threshold: Minimum similarity

        Returns:
            List of (document, score) tuples
        """
        # Build base query
        base_query = session.query(VectorDocument)

        # Apply filters
        filters = [~VectorDocument.is_deleted]

        if query.project:
            filters.append(VectorDocument.project == query.project)
        if query.namespace:
            filters.append(VectorDocument.namespace == query.namespace)

        # Apply custom filters from query
        if query.filters:
            for key, value in query.filters.items():
                if hasattr(VectorDocument, key):
                    filters.append(getattr(VectorDocument, key) == value)

        if filters:
            base_query = base_query.filter(and_(*filters))

        # For MySQL HeatWave, we'll use a custom function for vector similarity
        # This assumes HeatWave has a vector_similarity function
        # If not available, we'll fetch all and compute in Python

        try:
            # Try to use MySQL vector similarity function
            results = self._mysql_vector_search(
                session, base_query, query_vector, query.top_k, similarity_threshold
            )
        except Exception as e:
            logger.warning(f"MySQL vector search failed, falling back to Python: {e}")
            # Fallback to Python-based similarity computation
            results = self._python_vector_search(
                base_query, query_vector, query.top_k, similarity_threshold
            )

        return results

    def _mysql_vector_search(
        self,
        session: Session,
        base_query,
        query_vector: np.ndarray,
        top_k: int,
        similarity_threshold: float,
    ) -> list[tuple[VectorDocument, float]]:
        """Perform vector search using MySQL functions.

        This method attempts to use HeatWave's native vector functions.
        """
        # Convert query vector to string format for MySQL
        vector_str = f"[{','.join(map(str, query_vector.tolist()))}]"

        # Build SQL query with vector similarity
        # This assumes a VECTOR_SIMILARITY function exists in HeatWave
        sql = text("""
            SELECT
                id,
                VECTOR_SIMILARITY(vector, :query_vector) as similarity
            FROM vector_documents
            WHERE is_deleted = 0
                AND (:project IS NULL OR project = :project)
                AND (:namespace IS NULL OR namespace = :namespace)
            HAVING similarity >= :threshold
            ORDER BY similarity DESC
            LIMIT :limit
        """)

        # Execute query
        result = session.execute(
            sql,
            {
                "query_vector": vector_str,
                "project": base_query.project
                if hasattr(base_query, "project")
                else None,
                "namespace": base_query.namespace
                if hasattr(base_query, "namespace")
                else None,
                "threshold": similarity_threshold,
                "limit": top_k,
            },
        )

        # Fetch documents
        doc_scores = []
        for row in result:
            doc = session.query(VectorDocument).get(row.id)
            if doc:
                doc_scores.append((doc, row.similarity))

        return doc_scores

    def _python_vector_search(
        self,
        base_query,
        query_vector: np.ndarray,
        top_k: int,
        similarity_threshold: float,
    ) -> list[tuple[VectorDocument, float]]:
        """Perform vector search using Python computation.

        This is a fallback method when MySQL vector functions are not available.
        """
        # Fetch all matching documents
        documents = base_query.all()

        # Compute similarities
        doc_scores = []
        for doc in documents:
            if doc.vector is not None:
                # Compute cosine similarity
                similarity = self._cosine_similarity(query_vector, doc.vector)

                if similarity >= similarity_threshold:
                    doc_scores.append((doc, similarity))

        # Sort by similarity and take top k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _build_metadata(self, doc: VectorDocument) -> dict[str, Any]:
        """Build metadata dictionary from document."""
        metadata = doc.meta or {}

        # Add document fields to metadata
        metadata.update(
            {
                "source": doc.source,
                "source_id": doc.source_id,
                "document_title": doc.document_title,
                "author": doc.author,
                "tags": doc.tags,
                "lang": doc.lang,
                "project": doc.project,
                "namespace": doc.namespace,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "embedding_model": doc.embedding_model,
            }
        )

        return metadata

    def search_by_similarity(
        self,
        reference_text: str,
        top_k: int = 10,
        project: Optional[str] = None,
        namespace: Optional[str] = None,
        exclude_ids: Optional[list[int]] = None,
    ) -> list[VectorSearchResult]:
        """Find similar documents to a reference text.

        Args:
            reference_text: Reference text to find similar documents
            top_k: Number of results to return
            project: Filter by project
            namespace: Filter by namespace
            exclude_ids: Document IDs to exclude

        Returns:
            List of similar documents
        """
        query = VectorSearchQuery(
            query_text=reference_text,
            top_k=top_k,
            project=project,
            namespace=namespace,
            include_metadata=True,
        )

        results = self.search(query)

        # Filter out excluded IDs
        if exclude_ids:
            results = [r for r in results if r.id not in exclude_ids]

        return results[:top_k]

    def hybrid_search(
        self,
        query: VectorSearchQuery,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> list[VectorSearchResult]:
        """Perform hybrid search combining vector and keyword search.

        Args:
            query: Search query
            keyword_weight: Weight for keyword search
            vector_weight: Weight for vector search

        Returns:
            Combined search results
        """
        # Normalize weights
        total_weight = keyword_weight + vector_weight
        keyword_weight /= total_weight
        vector_weight /= total_weight

        # Perform vector search
        vector_results = self.search(query)

        # Perform keyword search
        keyword_results = self._keyword_search(query)

        # Combine results
        combined_scores = {}

        # Add vector search results
        for result in vector_results:
            combined_scores[result.id] = {
                "result": result,
                "score": result.score * vector_weight,
            }

        # Add keyword search results
        for result in keyword_results:
            if result.id in combined_scores:
                combined_scores[result.id]["score"] += result.score * keyword_weight
            else:
                combined_scores[result.id] = {
                    "result": result,
                    "score": result.score * keyword_weight,
                }

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(), key=lambda x: x["score"], reverse=True
        )

        # Update scores and return
        final_results = []
        for item in sorted_results[: query.top_k]:
            result = item["result"]
            result.score = item["score"]
            final_results.append(result)

        return final_results

    def _keyword_search(self, query: VectorSearchQuery) -> list[VectorSearchResult]:
        """Perform keyword-based search."""
        with self.db_manager.get_session() as session:
            # Build query
            base_query = session.query(VectorDocument)

            # Apply filters
            filters = [
                ~VectorDocument.is_deleted,
                VectorDocument.text.contains(query.query_text),
            ]

            if query.project:
                filters.append(VectorDocument.project == query.project)
            if query.namespace:
                filters.append(VectorDocument.namespace == query.namespace)

            base_query = base_query.filter(and_(*filters))

            # Get results
            documents = base_query.limit(query.top_k * 2).all()

            # Calculate simple relevance score
            results = []
            query_lower = query.query_text.lower()

            for doc in documents:
                # Simple scoring based on occurrence count
                text_lower = doc.text.lower()
                score = text_lower.count(query_lower) / len(text_lower.split())

                result = VectorSearchResult(
                    id=doc.id,
                    text=doc.text,
                    score=score,
                    source=doc.source,
                    chunk_index=doc.chunk_index,
                )

                if query.include_metadata:
                    result.meta = self._build_metadata(doc)

                results.append(result)

            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)

            return results[: query.top_k]
