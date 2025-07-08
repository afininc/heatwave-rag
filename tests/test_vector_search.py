"""Tests for vector search functionality."""

import numpy as np
import pytest

from heatwave_rag.embeddings import EmbeddingManager
from heatwave_rag.schemas import DocumentChunk, DocumentMetadata, VectorSearchQuery
from heatwave_rag.vector_search import VectorSearchEngine
from tests.test_embeddings import MockEmbeddings


class TestVectorSearch:
    """Test VectorSearchEngine class."""

    @pytest.fixture
    def search_engine(self, db_manager):
        """Vector search engine with mock embeddings."""
        mock_embeddings = MockEmbeddings()
        embedding_manager = EmbeddingManager(
            db_manager=db_manager,
            embeddings=mock_embeddings,
            embedding_model_name="mock-model",
        )
        return VectorSearchEngine(db_manager, embedding_manager)

    @pytest.fixture
    def sample_documents(self, search_engine, clean_database, test_table_name):
        """Create sample documents for testing."""
        from heatwave_rag.models import VectorDocument
        from heatwave_rag.schemas import TableInitConfig

        original_table_name = VectorDocument.__tablename__
        VectorDocument.__tablename__ = test_table_name

        try:
            # Initialize table
            table_config = TableInitConfig(
                table_name=test_table_name,
                vector_dimension=384,
            )
            search_engine.db_manager.init_tables(table_config)

            # Create diverse documents
            documents = [
                DocumentChunk(
                    text="Artificial Intelligence and machine learning",
                    metadata=DocumentMetadata(
                        source="doc1.txt",
                        project="ai_project",
                        namespace="tech",
                        tags=["AI", "ML"],
                    ),
                    chunk_index=0,
                ),
                DocumentChunk(
                    text="Deep learning neural networks",
                    metadata=DocumentMetadata(
                        source="doc2.txt",
                        project="ai_project",
                        namespace="tech",
                        tags=["DL", "NN"],
                    ),
                    chunk_index=0,
                ),
                DocumentChunk(
                    text="Natural language processing applications",
                    metadata=DocumentMetadata(
                        source="doc3.txt",
                        project="nlp_project",
                        namespace="tech",
                        tags=["NLP"],
                    ),
                    chunk_index=0,
                ),
                DocumentChunk(
                    text="Computer vision and image recognition",
                    metadata=DocumentMetadata(
                        source="doc4.txt",
                        project="cv_project",
                        namespace="tech",
                        tags=["CV", "Images"],
                    ),
                    chunk_index=0,
                ),
            ]

            # Store documents
            doc_ids = search_engine.embedding_manager.store_documents(documents)

            yield doc_ids
        finally:
            VectorDocument.__tablename__ = original_table_name

    def test_initialization(self, db_manager):
        """Test search engine initialization."""
        embedding_manager = EmbeddingManager(db_manager)
        engine = VectorSearchEngine(db_manager, embedding_manager)

        assert engine.db_manager == db_manager
        assert engine.embedding_manager == embedding_manager

    def test_basic_search(self, search_engine, sample_documents):
        """Test basic vector search."""
        query = VectorSearchQuery(
            query_text="machine learning algorithms",
            top_k=2,
        )

        results = search_engine.search(query)

        assert len(results) <= 2
        assert all(hasattr(r, "score") for r in results)
        assert all(hasattr(r, "text") for r in results)

    def test_search_with_filters(self, search_engine, sample_documents):
        """Test search with project filter."""
        query = VectorSearchQuery(
            query_text="artificial intelligence",
            top_k=10,
            project="ai_project",
        )

        results = search_engine.search(query)

        # Should only return results from ai_project
        assert all(r.metadata["project"] == "ai_project" for r in results)

    def test_search_with_namespace(self, search_engine, sample_documents):
        """Test search with namespace filter."""
        query = VectorSearchQuery(
            query_text="technology",
            top_k=10,
            namespace="tech",
        )

        results = search_engine.search(query)

        # All our test documents are in tech namespace
        assert len(results) > 0
        assert all(r.metadata["namespace"] == "tech" for r in results)

    def test_similarity_threshold(self, search_engine, sample_documents):
        """Test search with similarity threshold."""
        query = VectorSearchQuery(
            query_text="quantum computing",  # Unrelated query
            top_k=10,
        )

        # With high threshold, should get fewer results
        results_high = search_engine.search(query, similarity_threshold=0.9)
        results_low = search_engine.search(query, similarity_threshold=0.1)

        assert len(results_high) <= len(results_low)

    def test_cosine_similarity(self, search_engine):
        """Test cosine similarity calculation."""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        vec3 = np.array([0, 1, 0])
        vec4 = np.array([0.707, 0.707, 0])

        # Same vectors should have similarity 1
        assert search_engine._cosine_similarity(vec1, vec2) == pytest.approx(1.0)

        # Orthogonal vectors should have similarity 0
        assert search_engine._cosine_similarity(vec1, vec3) == pytest.approx(0.0)

        # 45-degree angle should have similarity ~0.707
        assert search_engine._cosine_similarity(vec1, vec4) == pytest.approx(
            0.707, rel=1e-3
        )

    def test_search_by_similarity(self, search_engine, sample_documents):
        """Test finding similar documents."""
        results = search_engine.search_by_similarity(
            reference_text="neural networks and deep learning",
            top_k=2,
        )

        assert len(results) <= 2
        assert all(hasattr(r, "score") for r in results)

    def test_hybrid_search(self, search_engine, sample_documents):
        """Test hybrid search combining vector and keyword."""
        query = VectorSearchQuery(
            query_text="learning",
            top_k=3,
        )

        results = search_engine.hybrid_search(
            query,
            keyword_weight=0.5,
            vector_weight=0.5,
        )

        assert len(results) <= 3
        # Documents containing "learning" should rank higher
        learning_docs = [r for r in results if "learning" in r.text.lower()]
        assert len(learning_docs) > 0

    def test_empty_search_results(self, search_engine, sample_documents):
        """Test search with no matches."""
        query = VectorSearchQuery(
            query_text="test query",
            top_k=10,
            project="non_existent_project",
        )

        results = search_engine.search(query)
        assert len(results) == 0

    def test_metadata_inclusion(self, search_engine, sample_documents):
        """Test that metadata is properly included in results."""
        query = VectorSearchQuery(
            query_text="artificial intelligence",
            top_k=5,
            include_metadata=True,
        )

        results = search_engine.search(query)

        assert len(results) > 0
        for result in results:
            assert result.metadata is not None
            assert "source" in result.metadata
            assert "tags" in result.metadata
