"""Document processing example for HeatWave RAG."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

import heatwave_rag

# Load environment variables
load_dotenv()


def main():
    """Document processing example."""
    # Initialize with HuggingFace embeddings (no API key required)
    rag = heatwave_rag.HeatWaveRAG(
        connection_config={
            "host": os.getenv("MYSQL_HOST", "localhost"),
            "port": int(os.getenv("MYSQL_PORT", "3306")),
            "user": os.getenv("MYSQL_USER"),
            "password": os.getenv("MYSQL_PASSWORD"),
            "database": os.getenv("MYSQL_DATABASE"),
        },
        embeddings=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        ),
        vector_dimension=384,  # all-MiniLM-L6-v2 produces 384-dim vectors
    )

    print("✅ HeatWave RAG initialized with HuggingFace embeddings")

    # Example 1: Process a single file
    print("\n📄 Processing a single file...")

    # Create a sample markdown file
    sample_dir = Path("./sample_docs")
    sample_dir.mkdir(exist_ok=True)

    sample_file = sample_dir / "ai_overview.md"
    sample_file.write_text("""
# Artificial Intelligence Overview

## What is AI?

Artificial Intelligence (AI) refers to the simulation of human intelligence in machines
that are programmed to think and learn like humans. AI systems can perform tasks that
typically require human intelligence, such as visual perception, speech recognition,
decision-making, and language translation.

## Types of AI

1. **Narrow AI**: Designed for specific tasks (e.g., voice assistants, recommendation systems)
2. **General AI**: Hypothetical AI with human-like general intelligence
3. **Super AI**: Theoretical AI that surpasses human intelligence

## Applications

- Healthcare: Disease diagnosis, drug discovery
- Finance: Fraud detection, algorithmic trading
- Transportation: Autonomous vehicles, traffic optimization
- Education: Personalized learning, automated grading
""")

    doc_ids = rag.add_documents(
        str(sample_file),
        metadata={
            "project": "ai_docs",
            "namespace": "educational",
            "document_type": "overview",
        },
        chunk_size=500,
        chunk_overlap=50,
    )

    print(f"Processed {sample_file.name}: {len(doc_ids)} chunks created")

    # Example 2: Process multiple files with different formats
    print("\n📁 Processing multiple files...")

    # Create more sample files
    (sample_dir / "ml_basics.txt").write_text("""
Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn
and improve from experience without being explicitly programmed. It focuses on developing
computer programs that can access data and use it to learn for themselves.

Key concepts:
- Supervised Learning: Learning from labeled data
- Unsupervised Learning: Finding patterns in unlabeled data
- Reinforcement Learning: Learning through trial and error
""")

    (sample_dir / "data.json").write_text("""
{
    "concepts": [
        {
            "name": "Neural Networks",
            "description": "Computing systems inspired by biological neural networks",
            "applications": ["Image recognition", "Natural language processing"]
        },
        {
            "name": "Deep Learning",
            "description": "Machine learning using artificial neural networks with multiple layers",
            "applications": ["Computer vision", "Speech recognition", "Machine translation"]
        }
    ]
}
""")

    # Process entire directory
    doc_ids = rag.add_directory(
        sample_dir,
        glob_pattern="**/*",
        metadata={
            "project": "ai_docs",
            "namespace": "educational",
        },
        chunk_size=300,
        exclude=["**/*.pyc", "**/__pycache__/**"],
    )

    print(f"Processed directory: {len(doc_ids)} total chunks")

    # Example 3: Custom text splitting
    print("\n✂️ Using custom text splitter...")

    # Get the document processor
    processor = rag.document_processor

    # Create a custom splitter for code
    code_splitter = processor.create_custom_splitter(
        splitter_type="python",
        chunk_size=200,
    )

    code_text = '''
def train_model(X, y, model_type="random_forest"):
    """Train a machine learning model."""
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == "svm":
        from sklearn.svm import SVC
        model = SVC(kernel="rbf")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    from sklearn.metrics import accuracy_score, classification_report

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    return {"accuracy": accuracy, "report": report}
'''

    chunks = processor.process_text(
        code_text,
        metadata=heatwave_rag.DocumentMetadata(
            source="example_code.py",
            lang="python",
            project="ai_docs",
            namespace="code_examples",
        ),
        splitter=code_splitter,
    )

    # Store the chunks
    doc_ids = rag.embedding_manager.store_documents(chunks)
    print(f"Processed Python code: {len(doc_ids)} chunks")

    # Example 4: Search with filters
    print("\n🔍 Searching with filters...")

    results = rag.search(
        "machine learning model training",
        top_k=5,
        project="ai_docs",
        filters={"namespace": "code_examples"},
    )

    print(f"Found {len(results)} results in code examples:")
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. Score: {result.score:.3f}")
        print(f"   Text preview: {result.text[:100]}...")
        print(f"   Source: {result.metadata.get('source', 'Unknown')}")

    # Cleanup
    print("\n🧹 Cleaning up...")

    # Delete test documents
    deleted = rag.delete_documents(project="ai_docs")
    print(f"Deleted {deleted} documents")

    # Remove sample directory
    import shutil

    shutil.rmtree(sample_dir)

    rag.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
