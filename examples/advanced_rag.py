"""Advanced RAG example with custom prompts and hybrid search."""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings

import heatwave_rag

# Load environment variables
load_dotenv()


def main():
    """Advanced RAG usage example."""
    # Initialize with Anthropic Claude for generation
    rag = heatwave_rag.HeatWaveRAG(
        connection_config={
            "host": os.getenv("MYSQL_HOST", "localhost"),
            "port": int(os.getenv("MYSQL_PORT", "3306")),
            "user": os.getenv("MYSQL_USER"),
            "password": os.getenv("MYSQL_PASSWORD"),
            "database": os.getenv("MYSQL_DATABASE"),
        },
        embeddings=OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        llm=ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-sonnet-20240229",
            temperature=0.3,
        ),
    )

    print("✅ HeatWave RAG initialized with Claude")

    # Prepare sample documents about RAG
    documents = [
        """
        Retrieval-Augmented Generation (RAG) is an AI framework that combines the power
        of retrieval systems with generative language models. RAG addresses the limitations
        of traditional language models by incorporating external knowledge sources.
        """,
        """
        The RAG process involves three main steps:
        1. Retrieval: Finding relevant documents from a knowledge base
        2. Augmentation: Combining retrieved context with the query
        3. Generation: Producing a response using the augmented input
        """,
        """
        Benefits of RAG include:
        - Access to up-to-date information beyond training data
        - Reduced hallucination through grounded responses
        - Ability to cite sources for transparency
        - Cost-effective compared to fine-tuning large models
        """,
        """
        Vector databases are crucial for RAG systems. They store document embeddings
        and enable efficient similarity search. Popular options include Pinecone,
        Weaviate, Chroma, and now Oracle HeatWave with vector support.
        """,
        """
        Chunking strategies affect RAG performance. Common approaches:
        - Fixed-size chunks with overlap
        - Semantic chunking based on content structure
        - Sliding window approach for maintaining context
        - Recursive splitting for hierarchical documents
        """,
    ]

    # Add documents with rich metadata
    print("\n📄 Adding RAG knowledge base...")

    for i, doc in enumerate(documents):
        rag.add_documents(
            [doc],
            metadata={
                "project": "rag_knowledge",
                "namespace": "technical",
                "topic": "rag_systems",
                "section_id": f"section_{i+1}",
                "importance": "high" if i < 3 else "medium",
            },
        )

    print(f"Added {len(documents)} documents about RAG")

    # Example 1: Custom prompt for technical explanation
    print("\n🎯 Using custom prompt for technical explanation...")

    technical_prompt = """You are a technical documentation expert.
Use the provided context to create a clear, structured explanation.

Context: {context}

Question: {question}

Please provide:
1. A brief overview
2. Key technical details
3. Practical implications

Answer:"""

    response = rag.query(
        "How do vector databases enhance RAG systems?",
        prompt_template=technical_prompt,
        project="rag_knowledge",
        top_k=3,
    )

    print(f"Technical Answer:\n{response['answer']}")

    # Example 2: Hybrid search combining vector and keyword
    print("\n🔄 Performing hybrid search...")

    hybrid_results = rag.vector_search_engine.hybrid_search(
        heatwave_rag.VectorSearchQuery(
            query_text="chunking strategies for documents",
            top_k=5,
            project="rag_knowledge",
        ),
        keyword_weight=0.4,
        vector_weight=0.6,
    )

    print("Hybrid search results (combining semantic + keyword matching):")
    for i, result in enumerate(hybrid_results, 1):
        print(f"\n{i}. Combined Score: {result.score:.3f}")
        print(f"   Text preview: {result.text.strip()[:100]}...")

    # Example 3: Streaming response
    print("\n📡 Streaming RAG response...")

    print("Question: What are the main components of a RAG system?")
    print("Answer: ", end="", flush=True)

    for chunk in rag.rag_engine.stream_query(
        "What are the main components of a RAG system?",
        top_k=3,
        project="rag_knowledge",
    ):
        print(chunk, end="", flush=True)
    print("\n")

    # Example 4: Multi-turn conversation with context
    print("\n💬 Multi-turn conversation...")

    conversation_prompt = """You are a helpful AI assistant in a conversation about RAG systems.
Use the context to provide accurate, conversational responses.

Previous context: {context}

User: {question}
Assistant: Be conversational but accurate!"""

    questions = [
        "What is RAG and why is it important?",
        "Can you elaborate on the retrieval component?",
        "How does HeatWave fit into this picture?",
    ]

    for q in questions:
        print(f"\n👤 User: {q}")
        response = rag.query(
            q,
            prompt_template=conversation_prompt,
            project="rag_knowledge",
            top_k=2,
        )
        print(f"🤖 Assistant: {response['answer']}")

    # Example 5: Similarity search for related documents
    print("\n🔗 Finding similar documents...")

    similar_docs = rag.vector_search_engine.search_by_similarity(
        reference_text="Vector databases store embeddings for similarity search",
        top_k=3,
        project="rag_knowledge",
    )

    print("Documents similar to reference text:")
    for i, doc in enumerate(similar_docs, 1):
        print(f"\n{i}. Similarity: {doc.score:.3f}")
        print(f"   Content: {doc.text.strip()[:150]}...")

    # Cleanup
    print("\n🧹 Cleaning up...")
    deleted = rag.delete_documents(project="rag_knowledge")
    print(f"Deleted {deleted} documents from the knowledge base")

    rag.close()
    print("\n✅ Advanced RAG example complete!")


if __name__ == "__main__":
    main()
