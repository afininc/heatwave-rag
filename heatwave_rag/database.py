"""Database connection and initialization utilities."""

import logging
from contextlib import contextmanager
from typing import Any, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from heatwave_rag.models import Base, VectorDocument
from heatwave_rag.schemas import ConnectionConfig, TableInitConfig

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self, config: ConnectionConfig):
        """Initialize database manager with connection configuration."""
        self.config = config
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

    @property
    def engine(self) -> Engine:
        """Get or create database engine."""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """Get or create session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory

    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine with connection pooling."""
        connection_url = (
            f"mysql+pymysql://{self.config.user}:{self.config.password}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
            f"?charset={self.config.charset}"
        )

        return create_engine(
            connection_url,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            echo=False,
        )

    @contextmanager
    def get_session(self) -> Session:
        """Context manager for database sessions."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def init_tables(self, table_config: TableInitConfig) -> None:
        """Initialize database tables."""
        if table_config.drop_if_exists:
            self._drop_tables(table_config.table_name)

        # Update the table name if different from default
        if table_config.table_name != "vector_documents":
            VectorDocument.__tablename__ = table_config.table_name

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create additional indexes if needed
        if table_config.create_indexes:
            self._create_custom_indexes(table_config)

        logger.info(f"Initialized table: {table_config.table_name}")

    def _drop_tables(self, table_name: str) -> None:
        """Drop existing tables."""
        with self.engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            conn.commit()

    def _create_custom_indexes(self, table_config: TableInitConfig) -> None:
        """Create custom indexes for better performance."""
        # Additional custom indexes can be added here
        pass

    def verify_table_compatibility(
        self, table_name: str = "vector_documents"
    ) -> dict[str, Any]:
        """Verify if existing table is compatible with our schema."""
        with self.engine.connect() as conn:
            # Check if table exists
            result = conn.execute(
                text(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema = :db AND table_name = :table"
                ),
                {"db": self.config.database, "table": table_name},
            )

            if result.scalar() == 0:
                return {
                    "compatible": False,
                    "exists": False,
                    "message": f"Table {table_name} does not exist",
                }

            # Check columns
            result = conn.execute(
                text(
                    "SELECT column_name, data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_schema = :db AND table_name = :table"
                ),
                {"db": self.config.database, "table": table_name},
            )

            existing_columns = {
                row[0]: {"type": row[1], "nullable": row[2]} for row in result
            }

            # Required columns
            required_columns = {
                "id": {"type": "int", "nullable": "NO"},
                "vector": {"type": ["longtext", "vector"], "nullable": "NO"},
                "text": {"type": "text", "nullable": "NO"},
                "metadata": {"type": "json", "nullable": "YES"},
                "source": {"type": "varchar", "nullable": "YES"},
                "source_id": {"type": "varchar", "nullable": "YES"},
                "chunk_index": {"type": "int", "nullable": "NO"},
                "created_at": {"type": "datetime", "nullable": "NO"},
                "document_title": {"type": "varchar", "nullable": "YES"},
                "author": {"type": "varchar", "nullable": "YES"},
                "tags": {"type": "json", "nullable": "YES"},
                "lang": {"type": "varchar", "nullable": "YES"},
                "project": {"type": "varchar", "nullable": "YES"},
                "namespace": {"type": "varchar", "nullable": "YES"},
                "is_deleted": {"type": "tinyint", "nullable": "NO"},
                "embedding_model": {"type": "varchar", "nullable": "NO"},
            }

            missing_columns = []
            incompatible_columns = []

            for col_name, col_info in required_columns.items():
                if col_name not in existing_columns:
                    missing_columns.append(col_name)
                else:
                    existing_type = existing_columns[col_name]["type"]
                    expected_types = (
                        col_info["type"]
                        if isinstance(col_info["type"], list)
                        else [col_info["type"]]
                    )

                    if existing_type not in expected_types:
                        incompatible_columns.append(
                            {
                                "column": col_name,
                                "expected": expected_types,
                                "actual": existing_type,
                            }
                        )

            if missing_columns or incompatible_columns:
                return {
                    "compatible": False,
                    "exists": True,
                    "missing_columns": missing_columns,
                    "incompatible_columns": incompatible_columns,
                    "message": "Table structure is not compatible",
                }

            return {
                "compatible": True,
                "exists": True,
                "message": "Table is compatible",
            }

    def close(self) -> None:
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
