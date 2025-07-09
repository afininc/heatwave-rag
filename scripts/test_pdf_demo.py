"""Test HeatWave RAG with PDF documents from demos folder."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import heatwave_rag

# Load environment variables
load_dotenv()


def main():
    """Test HeatWave RAG with PDF documents."""
    print("🚀 Starting HeatWave RAG PDF Demo Test\n")

    # Initialize HeatWave RAG
    print("📋 Initializing HeatWave RAG...")
    rag = heatwave_rag.HeatWaveRAG(
        connection_config={
            "host": os.getenv("MYSQL_HOST"),
            "port": int(os.getenv("MYSQL_PORT")),
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
            model="gpt-4",
            temperature=0.7,
        ),
    )
    print("✅ HeatWave RAG initialized successfully!\n")

    # Process PDF files from demos folder
    print("📂 Processing PDF files from demos folder...")
    demos_path = Path("./demos")

    # Add all PDF files
    doc_ids = rag.add_directory(
        demos_path,
        glob_pattern="*.pdf",
        metadata={
            "project": "rsupport_demo",
            "namespace": "pdfs",
            "document_type": "pdf",
        },
        chunk_size=1000,
        chunk_overlap=200,
    )

    print(f"✅ Processed {len(doc_ids)} chunks from PDF files\n")

    # Get document count
    count = rag.get_document_count(project="rsupport_demo")
    print(f"📊 Total documents in database: {count}\n")

    # Test queries
    test_queries = [
        "RSupport가 무엇인가요?",
        "주요 제품이나 서비스는 무엇인가요?",
        "원격 지원 솔루션에 대해 알려주세요",
    ]

    print("🔍 Testing RAG queries:\n")

    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Query: {query}")
        print("-" * 50)

        # Search for relevant documents
        search_results = rag.search(
            query,
            top_k=3,
            project="rsupport_demo",
        )

        print(f"Found {len(search_results)} relevant chunks")

        # Perform RAG query
        response = rag.query(
            query,
            top_k=5,
            project="rsupport_demo",
            return_sources=True,
        )

        print(f"\nAnswer: {response['answer']}")

        if response.get("sources"):
            print("\nTop sources:")
            for j, source in enumerate(response["sources"][:2], 1):
                print(f"  {j}. {source['text'][:150]}...")
                print(f"     Score: {source['score']:.3f}")

        print("\n" + "=" * 70 + "\n")

    # Test hybrid search
    print("🔄 Testing hybrid search...")
    hybrid_results = rag.vector_search_engine.hybrid_search(
        heatwave_rag.VectorSearchQuery(
            query_text="remote support",
            top_k=5,
            project="rsupport_demo",
        ),
        keyword_weight=0.3,
        vector_weight=0.7,
    )

    print(f"Found {len(hybrid_results)} results with hybrid search")
    for i, result in enumerate(hybrid_results[:3], 1):
        print(f"\n{i}. Score: {result.score:.3f}")
        print(f"   Text: {result.text[:100]}...")

    # Cleanup
    print("\n🧹 Cleaning up...")
    deleted = rag.delete_documents(project="rsupport_demo")
    print(f"Deleted {deleted} documents from the database")

    rag.close()
    print("\n✅ Demo test completed successfully!")


if __name__ == "__main__":
    main()
