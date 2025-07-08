"""Tests for embedding functionality."""


import pytest

from heatwave_rag.embeddings import EmbeddingManager
from heatwave_rag.schemas import DocumentChunk, DocumentMetadata


class MockEmbeddings:
    """Mock embeddings class for testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        """Generate mock embedding for a query."""
        # Simple hash-based fake embedding
        hash_val = hash(text) % 1000
        return [float(hash_val + i) / 1000 for i in range(self.dimension)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for documents."""
        return [self.embed_query(text) for text in texts]


class TestEmbeddingManager:
    """Test EmbeddingManager class."""

    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings instance."""
        return MockEmbeddings()

    @pytest.fixture
    def embedding_manager(self, db_manager, mock_embeddings):
        """Embedding manager with mock embeddings."""
        return EmbeddingManager(
            db_manager=db_manager,
            embeddings=mock_embeddings,
            embedding_model_name="mock-model",
        )

    def test_initialization(self, db_manager):
        """Test embedding manager initialization."""
        manager = EmbeddingManager(db_manager)
        assert manager.db_manager == db_manager
        assert manager.embeddings is None
        assert manager.embedding_model_name == "unknown"

    def test_set_embeddings(self, embedding_manager, mock_embeddings):
        """Test setting embeddings instance."""
        new_embeddings = MockEmbeddings(dimension=768)
        embedding_manager.set_embeddings(new_embeddings, "new-model")

        assert embedding_manager.embeddings == new_embeddings
        assert embedding_manager.embedding_model_name == "new-model"

    def test_embed_text_single(self, embedding_manager):
        """Test embedding single text."""
        text = "Test text"
        embedding = embedding_manager.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_text_multiple(self, embedding_manager):
        """Test embedding multiple texts."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = embedding_manager.embed_text(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)

    def test_embed_text_no_embeddings(self, db_manager):
        """Test embedding without embeddings instance."""
        manager = EmbeddingManager(db_manager)

        with pytest.raises(ValueError, match="No embeddings instance configured"):
            manager.embed_text("Test text")

    def test_store_documents(self, embedding_manager, sample_metadata, clean_database, test_table_name):
        """Test storing documents with embeddings."""
        # Create test chunks
        chunks = [
            DocumentChunk(
                text=f"Test chunk {i}",
                metadata=DocumentMetadata(**sample_metadata),
                chunk_index=i,
            )
            for i in range(3)
        ]

        # Initialize test table
        from heatwave_rag.models import VectorDocument
        from heatwave_rag.schemas import TableInitConfig

        original_table_name = VectorDocument.__tablename__
        VectorDocument.__tablename__ = test_table_name

        try:
            table_config = TableInitConfig(
                table_name=test_table_name,
                vector_dimension=384,
            )
            embedding_manager.db_manager.init_tables(table_config)

            # Store documents
            doc_ids = embedding_manager.store_documents(chunks)

            assert len(doc_ids) == 3
            assert all(isinstance(id, int) for id in doc_ids)

            # Verify documents were stored
            with embedding_manager.db_manager.get_session() as session:
                count = session.query(VectorDocument).count()
                assert count == 3
        finally:
            VectorDocument.__tablename__ = original_table_name

    def test_document_deduplication(self, embedding_manager, sample_metadata, clean_database, test_table_name):
        """Test that duplicate documents are skipped."""
        # Create duplicate chunks
        chunks = [
            DocumentChunk(
                text="Duplicate text",
                metadata=DocumentMetadata(**{**sample_metadata, "source_id": "dup_001"}),
                chunk_index=0,
            )
            for _ in range(3)
        ]

        from heatwave_rag.models import VectorDocument
        from heatwave_rag.schemas import TableInitConfig

        original_table_name = VectorDocument.__tablename__
        VectorDocument.__tablename__ = test_table_name

        try:
            table_config = TableInitConfig(
                table_name=test_table_name,
                vector_dimension=384,
            )
            embedding_manager.db_manager.init_tables(table_config)

            # Store documents with skip_existing=True
            doc_ids = embedding_manager.store_documents(chunks, skip_existing=True)

            # Only one should be stored
            assert len(doc_ids) == 1
        finally:
            VectorDocument.__tablename__ = original_table_name

    def test_get_document_count(self, embedding_manager, clean_database, test_table_name):
        """Test getting document count."""
        from heatwave_rag.models import VectorDocument
        from heatwave_rag.schemas import TableInitConfig

        original_table_name = VectorDocument.__tablename__
        VectorDocument.__tablename__ = test_table_name

        try:
            table_config = TableInitConfig(
                table_name=test_table_name,
                vector_dimension=384,
            )
            embedding_manager.db_manager.init_tables(table_config)

            # Initially should be 0
            count = embedding_manager.get_document_count()
            assert count == 0

            # Add some documents
            chunks = [
                DocumentChunk(
                    text=f"Test {i}",
                    metadata=DocumentMetadata(project="test", namespace="ns1"),
                    chunk_index=i,
                )
                for i in range(5)
            ]
            embedding_manager.store_documents(chunks)

            # Check count
            count = embedding_manager.get_document_count()
            assert count == 5

            # Check filtered count
            count = embedding_manager.get_document_count(project="test")
            assert count == 5

            count = embedding_manager.get_document_count(project="other")
            assert count == 0
        finally:
            VectorDocument.__tablename__ = original_table_name
