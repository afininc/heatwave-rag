"""Basic usage example for HeatWave RAG."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import heatwave_rag

# Load environment variables
load_dotenv()


def main():
    """Basic usage example."""
    # Initialize HeatWave RAG with connection configuration
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
            model="text-embedding-ada-002",
        ),
        llm=ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",
            temperature=0.7,
        ),
    )

    print("✅ HeatWave RAG initialized successfully!")

    # Example 1: Add text documents
    print("\n📄 Adding text documents...")

    texts = [
        "Artificial Intelligence (AI) is transforming industries across the globe.",
        "Machine learning algorithms can identify patterns in large datasets.",
        "Natural Language Processing enables computers to understand human language.",
    ]

    doc_ids = rag.add_documents(
        texts,
        metadata={
            "project": "ai_basics",
            "namespace": "tutorials",
            "tags": ["AI", "ML", "NLP"],
        },
    )

    print(f"Added {len(doc_ids)} documents")

    # Example 2: Search for similar documents
    print("\n🔍 Searching for similar documents...")

    search_results = rag.search(
        "What is machine learning?",
        top_k=3,
        project="ai_basics",
    )

    for i, result in enumerate(search_results, 1):
        print(f"\n{i}. Score: {result.score:.3f}")
        print(f"   Text: {result.text}")
        if result.metadata:
            print(f"   Tags: {result.metadata.get('tags', [])}")

    # Example 3: Perform RAG query
    print("\n🤖 Performing RAG query...")

    response = rag.query(
        "Explain the relationship between AI, ML, and NLP",
        top_k=3,
        project="ai_basics",
        return_sources=True,
    )

    print(f"\nAnswer: {response['answer']}")

    if response.get("sources"):
        print("\nSources:")
        for i, source in enumerate(response["sources"], 1):
            print(f"{i}. {source['text']}")
            print(f"   Score: {source['score']:.3f}")

    # Example 4: Get document count
    count = rag.get_document_count(project="ai_basics")
    print(f"\n📊 Total documents in 'ai_basics' project: {count}")

    # Close the connection
    rag.close()
    print("\n👋 Connection closed")


if __name__ == "__main__":
    main()
