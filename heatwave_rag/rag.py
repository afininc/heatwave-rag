"""RAG (Retrieval-Augmented Generation) functionality."""

import logging
from typing import Any, Optional

from langchain.chains import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate

from heatwave_rag.schemas import VectorSearchQuery, VectorSearchResult
from heatwave_rag.vector_search import VectorSearchEngine

logger = logging.getLogger(__name__)


class RAGEngine:
    """Performs RAG operations with vector search and LLM generation."""

    DEFAULT_RAG_PROMPT = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(
        self,
        vector_search_engine: VectorSearchEngine,
        llm: Optional[BaseChatModel] = None,
        prompt_template: Optional[str] = None,
    ):
        """Initialize RAG engine.

        Args:
            vector_search_engine: Vector search engine instance
            llm: LangChain LLM instance for generation
            prompt_template: Custom prompt template
        """
        self.vector_search_engine = vector_search_engine
        self.llm = llm
        self.prompt_template = prompt_template or self.DEFAULT_RAG_PROMPT

    def set_llm(self, llm: BaseChatModel) -> None:
        """Set the LLM instance.

        Args:
            llm: LangChain LLM instance
        """
        self.llm = llm

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        project: Optional[str] = None,
        namespace: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> list[VectorSearchResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve
            project: Filter by project
            namespace: Filter by namespace
            filters: Additional filters
            include_metadata: Include metadata in results

        Returns:
            List of retrieved documents
        """
        search_query = VectorSearchQuery(
            query_text=query,
            top_k=top_k,
            project=project,
            namespace=namespace,
            filters=filters,
            include_metadata=include_metadata,
        )

        results = self.vector_search_engine.search(search_query)
        logger.info(f"Retrieved {len(results)} documents for query: {query}")

        return results

    def format_context(
        self, results: list[VectorSearchResult], include_metadata: bool = False
    ) -> str:
        """Format search results into context string.

        Args:
            results: Search results
            include_metadata: Include metadata in context

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, result in enumerate(results, 1):
            context_part = f"[{i}] {result.text}"

            if include_metadata and result.metadata:
                # Add relevant metadata
                if result.metadata.get("source"):
                    context_part += f"\n(Source: {result.metadata['source']})"
                if result.metadata.get("document_title"):
                    context_part += f"\n(Title: {result.metadata['document_title']})"

            context_parts.append(context_part)

        return "\n\n".join(context_parts)

    def generate(
        self,
        query: str,
        context: str,
        prompt_template: Optional[str] = None,
        **llm_kwargs,
    ) -> str:
        """Generate answer using LLM with context.

        Args:
            query: User query
            context: Retrieved context
            prompt_template: Custom prompt template
            **llm_kwargs: Additional arguments for LLM

        Returns:
            Generated answer
        """
        if not self.llm:
            raise ValueError("No LLM instance configured")

        # Use custom template if provided
        template = prompt_template or self.prompt_template

        # Create prompt
        prompt = ChatPromptTemplate.from_template(template)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Generate response
        response = chain.run(question=query, context=context, **llm_kwargs)
        logger.info(f"Generated response for query: {query}")

        return response

    def query(
        self,
        query: str,
        top_k: int = 5,
        project: Optional[str] = None,
        namespace: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        include_metadata_in_context: bool = False,
        prompt_template: Optional[str] = None,
        return_sources: bool = False,
        **llm_kwargs,
    ) -> dict[str, Any]:
        """Perform full RAG query: retrieve and generate.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            project: Filter by project
            namespace: Filter by namespace
            filters: Additional filters
            include_metadata_in_context: Include metadata in context
            prompt_template: Custom prompt template
            return_sources: Return source documents
            **llm_kwargs: Additional arguments for LLM

        Returns:
            Dictionary with answer and optionally sources
        """
        # Retrieve relevant documents
        results = self.retrieve(
            query=query,
            top_k=top_k,
            project=project,
            namespace=namespace,
            filters=filters,
        )

        # Format context
        context = self.format_context(results, include_metadata_in_context)

        # Generate answer
        answer = self.generate(
            query=query,
            context=context,
            prompt_template=prompt_template,
            **llm_kwargs,
        )

        # Prepare response
        response = {"answer": answer, "query": query}

        if return_sources:
            response["sources"] = [
                {
                    "text": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                for result in results
            ]

        return response

    def stream_query(
        self,
        query: str,
        top_k: int = 5,
        project: Optional[str] = None,
        namespace: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        include_metadata_in_context: bool = False,
        prompt_template: Optional[str] = None,
        **llm_kwargs,
    ):
        """Perform RAG query with streaming response.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            project: Filter by project
            namespace: Filter by namespace
            filters: Additional filters
            include_metadata_in_context: Include metadata in context
            prompt_template: Custom prompt template
            **llm_kwargs: Additional arguments for LLM

        Yields:
            Chunks of generated response
        """
        if not self.llm:
            raise ValueError("No LLM instance configured")

        # Retrieve relevant documents
        results = self.retrieve(
            query=query,
            top_k=top_k,
            project=project,
            namespace=namespace,
            filters=filters,
        )

        # Format context
        context = self.format_context(results, include_metadata_in_context)

        # Use custom template if provided
        template = prompt_template or self.prompt_template

        # Create prompt
        prompt = ChatPromptTemplate.from_template(template)

        # Format the prompt
        formatted_prompt = prompt.format(question=query, context=context)

        # Stream response
        for chunk in self.llm.stream(formatted_prompt, **llm_kwargs):
            yield chunk.content

    def create_custom_prompt(
        self,
        system_message: Optional[str] = None,
        user_template: Optional[str] = None,
        include_examples: bool = False,
    ) -> str:
        """Create a custom prompt template.

        Args:
            system_message: System message for the prompt
            user_template: User message template
            include_examples: Include example Q&A pairs

        Returns:
            Custom prompt template string
        """
        parts = []

        if system_message:
            parts.append(f"System: {system_message}")

        if include_examples:
            parts.append("""Examples:
Context: The sky is blue during clear weather.
Question: What color is the sky?
Answer: The sky is blue during clear weather.

Context: Water freezes at 0 degrees Celsius.
Question: At what temperature does water freeze?
Answer: Water freezes at 0 degrees Celsius.""")

        if user_template:
            parts.append(user_template)
        else:
            parts.append(self.DEFAULT_RAG_PROMPT)

        return "\n\n".join(parts)
