"""Document processing utilities with LangChain integration."""

import logging
from typing import Any, Optional

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from langchain.text_splitter import (
    CharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)

from heatwave_rag.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process documents using LangChain loaders and splitters."""

    # Default file type to loader mapping
    DEFAULT_LOADERS = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".md": UnstructuredMarkdownLoader,
        ".json": JSONLoader,
        ".csv": CSVLoader,
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        custom_loaders: Optional[dict[str, BaseLoader]] = None,
        custom_splitter: Optional[Any] = None,
    ):
        """Initialize document processor.

        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
            custom_loaders: Custom file type to loader mapping
            custom_splitter: Custom text splitter
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Merge custom loaders with defaults
        self.loaders = self.DEFAULT_LOADERS.copy()
        if custom_loaders:
            self.loaders.update(custom_loaders)

        # Set up text splitter
        self.splitter = custom_splitter or RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def load_document(
        self, file_path: str, loader: Optional[BaseLoader] = None, **loader_kwargs
    ) -> list[Document]:
        """Load a document from file.

        Args:
            file_path: Path to the document
            loader: Custom loader instance
            **loader_kwargs: Additional arguments for the loader

        Returns:
            List of loaded documents
        """
        if loader:
            doc_loader = loader
        else:
            # Determine loader based on file extension
            file_ext = file_path.lower().split(".")[-1]
            file_ext = f".{file_ext}"

            loader_class = self.loaders.get(file_ext)
            if not loader_class:
                raise ValueError(f"No loader available for file type: {file_ext}")

            # Special handling for JSON loader
            if loader_class == JSONLoader and "jq_schema" not in loader_kwargs:
                loader_kwargs["jq_schema"] = "."

            doc_loader = loader_class(file_path, **loader_kwargs)

        try:
            documents = doc_loader.load()
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            raise

    def load_directory(
        self,
        directory_path: str,
        glob: str = "**/*",
        recursive: bool = True,
        exclude: Optional[list[str]] = None,
        **loader_kwargs,
    ) -> list[Document]:
        """Load all documents from a directory.

        Args:
            directory_path: Path to the directory
            glob: Glob pattern for file matching
            recursive: Whether to search recursively
            exclude: List of patterns to exclude
            **loader_kwargs: Additional arguments for loaders

        Returns:
            List of loaded documents
        """
        all_documents = []

        for file_ext, loader_class in self.loaders.items():
            try:
                loader = DirectoryLoader(
                    directory_path,
                    glob=f"**/*{file_ext}" if recursive else f"*{file_ext}",
                    loader_cls=loader_class,
                    loader_kwargs=loader_kwargs,
                    recursive=recursive,
                    exclude=exclude or [],
                )
                documents = loader.load()
                all_documents.extend(documents)
                logger.info(
                    f"Loaded {len(documents)} {file_ext} files from {directory_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to load {file_ext} files: {e}")

        return all_documents

    def split_documents(
        self, documents: list[Document], splitter: Optional[Any] = None
    ) -> list[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents to split
            splitter: Custom text splitter

        Returns:
            List of document chunks
        """
        text_splitter = splitter or self.splitter
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def process_file(
        self,
        file_path: str,
        metadata: Optional[dict[str, Any]] = None,
        loader: Optional[BaseLoader] = None,
        splitter: Optional[Any] = None,
        **loader_kwargs,
    ) -> list[DocumentChunk]:
        """Process a file: load and split into chunks.

        Args:
            file_path: Path to the file
            metadata: Additional metadata to add to chunks
            loader: Custom loader
            splitter: Custom splitter
            **loader_kwargs: Additional loader arguments

        Returns:
            List of document chunks
        """
        # Load document
        documents = self.load_document(file_path, loader, **loader_kwargs)

        # Add metadata
        if metadata:
            metadata_dict = metadata
            for doc in documents:
                doc.metadata.update(metadata_dict)
                doc.metadata["source"] = file_path

        # Split into chunks
        chunks = self.split_documents(documents, splitter)

        # Convert to DocumentChunk objects
        result = []
        for idx, chunk in enumerate(chunks):
            result.append(
                DocumentChunk(
                    text=chunk.page_content,
                    meta=chunk.metadata,
                    chunk_index=idx,
                )
            )

        return result

    def process_text(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        splitter: Optional[Any] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> list[DocumentChunk]:
        """Process raw text: split into chunks.

        Args:
            text: Raw text to process
            metadata: Metadata to add to chunks
            splitter: Custom text splitter

        Returns:
            List of document chunks
        """
        # Create document
        doc_metadata = metadata if metadata else {}
        document = Document(page_content=text, metadata=doc_metadata)

        # Split into chunks
        if not splitter and (chunk_size or chunk_overlap):
            # Create a temporary splitter with custom parameters
            temp_splitter = CharacterTextSplitter(
                chunk_size=chunk_size or self.chunk_size,
                chunk_overlap=chunk_overlap or self.chunk_overlap,
                separator="\n\n",
            )
            chunks = self.split_documents([document], temp_splitter)
        else:
            chunks = self.split_documents([document], splitter)

        # Convert to DocumentChunk objects
        result = []
        for idx, chunk in enumerate(chunks):
            result.append(
                DocumentChunk(
                    text=chunk.page_content,
                    meta=chunk.metadata,
                    chunk_index=idx,
                )
            )

        return result

    def create_custom_splitter(
        self, splitter_type: str = "recursive", **splitter_kwargs
    ) -> Any:
        """Create a custom text splitter.

        Args:
            splitter_type: Type of splitter
            **splitter_kwargs: Arguments for the splitter

        Returns:
            Text splitter instance
        """
        splitter_map = {
            "recursive": RecursiveCharacterTextSplitter,
            "character": CharacterTextSplitter,
            "token": TokenTextSplitter,
            "markdown": MarkdownTextSplitter,
            "python": PythonCodeTextSplitter,
        }

        splitter_class = splitter_map.get(splitter_type)
        if not splitter_class:
            raise ValueError(f"Unknown splitter type: {splitter_type}")

        # Set default chunk size and overlap if not provided
        if "chunk_size" not in splitter_kwargs:
            splitter_kwargs["chunk_size"] = self.chunk_size
        if "chunk_overlap" not in splitter_kwargs and splitter_type != "python":
            splitter_kwargs["chunk_overlap"] = self.chunk_overlap

        return splitter_class(**splitter_kwargs)
