"""Tests for database functionality."""


from heatwave_rag.database import DatabaseManager
from heatwave_rag.models import VectorDocument
from heatwave_rag.schemas import ConnectionConfig, TableInitConfig


class TestDatabaseManager:
    """Test DatabaseManager class."""

    def test_connection_creation(self, db_config: ConnectionConfig):
        """Test database connection creation."""
        manager = DatabaseManager(db_config)
        assert manager.engine is not None
        assert manager.session_factory is not None
        manager.close()

    def test_table_initialization(self, db_manager: DatabaseManager, test_table_name: str, clean_database):
        """Test table initialization."""
        table_config = TableInitConfig(
            table_name=test_table_name,
            vector_dimension=1536,
            create_indexes=True,
        )

        # Update table name for the model
        original_table_name = VectorDocument.__tablename__
        VectorDocument.__tablename__ = test_table_name

        try:
            db_manager.init_tables(table_config)

            # Verify table exists
            result = db_manager.verify_table_compatibility(test_table_name)
            assert result["exists"] is True
            assert result["compatible"] is True
        finally:
            # Restore original table name
            VectorDocument.__tablename__ = original_table_name

    def test_table_compatibility_check(self, db_manager: DatabaseManager):
        """Test table compatibility verification."""
        # Check non-existent table
        result = db_manager.verify_table_compatibility("non_existent_table")
        assert result["exists"] is False
        assert result["compatible"] is False

    def test_session_management(self, db_manager: DatabaseManager):
        """Test session context manager."""
        with db_manager.get_session() as session:
            assert session is not None
            assert session.is_active

    def test_drop_tables(self, db_manager: DatabaseManager, test_table_name: str, clean_database):
        """Test dropping tables."""
        table_config = TableInitConfig(
            table_name=test_table_name,
            drop_if_exists=True,
        )

        # Update table name for the model
        original_table_name = VectorDocument.__tablename__
        VectorDocument.__tablename__ = test_table_name

        try:
            # Create table first
            db_manager.init_tables(table_config)

            # Verify it exists
            result = db_manager.verify_table_compatibility(test_table_name)
            assert result["exists"] is True

            # Drop and recreate
            table_config.drop_if_exists = True
            db_manager.init_tables(table_config)

            # Should still exist
            result = db_manager.verify_table_compatibility(test_table_name)
            assert result["exists"] is True
        finally:
            # Restore original table name
            VectorDocument.__tablename__ = original_table_name
