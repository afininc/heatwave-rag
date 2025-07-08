"""Tests for document processing functionality."""

import tempfile
from pathlib import Path

import pytest

from heatwave_rag.document_processor import DocumentProcessor
from heatwave_rag.schemas import DocumentChunk, DocumentMetadata


class TestDocumentProcessor:
    """Test DocumentProcessor class."""

    def test_initialization(self):
        """Test document processor initialization."""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 50
        assert processor.splitter is not None

    def test_process_text(self, sample_text: str, sample_metadata: dict):
        """Test processing raw text."""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
        metadata = DocumentMetadata(**sample_metadata)

        chunks = processor.process_text(sample_text, metadata)

        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(len(chunk.text) <= 200 for chunk in chunks)

        # Check metadata propagation
        for chunk in chunks:
            assert chunk.metadata.source == sample_metadata["source"]
            assert chunk.metadata.author == sample_metadata["author"]

    def test_process_file(self, sample_text: str, sample_metadata: dict):
        """Test processing a text file."""
        processor = DocumentProcessor()
        metadata = DocumentMetadata(**sample_metadata)

        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(sample_text)
            temp_path = f.name

        try:
            chunks = processor.process_file(temp_path, metadata)

            assert len(chunks) > 0
            assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
            assert chunks[0].metadata.source == temp_path
        finally:
            Path(temp_path).unlink()

    def test_chunk_indexing(self, sample_text: str):
        """Test that chunks are properly indexed."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        chunks = processor.process_text(sample_text)

        # Check chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_custom_splitter(self, sample_text: str):
        """Test using custom text splitter."""
        processor = DocumentProcessor()

        # Create custom splitter
        custom_splitter = processor.create_custom_splitter(
            splitter_type="character",
            chunk_size=150,
            chunk_overlap=15,
            separator="\n",
        )

        chunks = processor.process_text(sample_text, splitter=custom_splitter)
        assert len(chunks) > 0

    def test_empty_text_handling(self):
        """Test handling of empty text."""
        processor = DocumentProcessor()
        chunks = processor.process_text("")

        assert len(chunks) == 1
        assert chunks[0].text == ""
        assert chunks[0].chunk_index == 0

    @pytest.mark.parametrize(
        "splitter_type",
        [
            "recursive",
            "character",
            "token",
            "markdown",
            "python",
        ],
    )
    def test_different_splitters(self, splitter_type: str, sample_text: str):
        """Test different splitter types."""
        processor = DocumentProcessor()

        splitter = processor.create_custom_splitter(
            splitter_type=splitter_type,
            chunk_size=200,
        )

        chunks = processor.process_text(sample_text, splitter=splitter)
        assert len(chunks) > 0
