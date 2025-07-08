"""Pytest configuration and fixtures."""

import os
from collections.abc import Generator

import pytest
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from heatwave_rag.database import DatabaseManager
from heatwave_rag.schemas import ConnectionConfig

# Load environment variables
load_dotenv()


@pytest.fixture(scope="session")
def db_config() -> ConnectionConfig:
    """Database configuration from environment variables."""
    return ConnectionConfig(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "heatwave_rag_test"),
    )


@pytest.fixture(scope="session")
def db_manager(db_config: ConnectionConfig) -> DatabaseManager:
    """Database manager instance."""
    manager = DatabaseManager(db_config)
    yield manager
    manager.close()


@pytest.fixture(scope="function")
def db_session(db_manager: DatabaseManager) -> Generator[Session, None, None]:
    """Database session for tests."""
    with db_manager.get_session() as session:
        yield session


@pytest.fixture(scope="session")
def test_table_name() -> str:
    """Test table name."""
    return "test_vector_documents"


@pytest.fixture(scope="function")
def clean_database(db_manager: DatabaseManager, test_table_name: str):
    """Clean database before and after tests."""
    # Drop test tables before test
    with db_manager.engine.connect() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {test_table_name}")
        conn.commit()

    yield

    # Drop test tables after test
    with db_manager.engine.connect() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {test_table_name}")
        conn.commit()


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing."""
    return """
    Artificial Intelligence (AI) is revolutionizing the way we live and work.
    Machine learning algorithms can now process vast amounts of data to identify
    patterns and make predictions with remarkable accuracy. Deep learning, a subset
    of machine learning, uses neural networks to solve complex problems in areas
    like computer vision, natural language processing, and speech recognition.
    """


@pytest.fixture
def sample_metadata() -> dict:
    """Sample metadata for testing."""
    return {
        "source": "test_document.txt",
        "source_id": "test_001",
        "document_title": "AI Overview",
        "author": "Test Author",
        "tags": ["AI", "ML", "technology"],
        "lang": "en",
        "project": "test_project",
        "namespace": "test_namespace",
    }
