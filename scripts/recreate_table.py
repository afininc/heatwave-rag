"""Recreate table with LONGTEXT for vector column."""

import os
from dotenv import load_dotenv
import pymysql

# Load environment variables
load_dotenv()

# Connect to MySQL
conn = pymysql.connect(
    host=os.getenv("MYSQL_HOST"),
    port=int(os.getenv("MYSQL_PORT")),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE"),
)

try:
    with conn.cursor() as cursor:
        # Drop existing table
        cursor.execute("DROP TABLE IF EXISTS vector_documents")
        print("✅ Dropped existing table")
        
        # Create new table with LONGTEXT for vector
        cursor.execute("""
        CREATE TABLE vector_documents (
            id INTEGER NOT NULL AUTO_INCREMENT,
            vector LONGTEXT NOT NULL,
            text TEXT NOT NULL,
            metadata JSON,
            source VARCHAR(500),
            source_id VARCHAR(255),
            chunk_index INTEGER NOT NULL DEFAULT 0,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            document_title VARCHAR(500),
            author VARCHAR(255),
            tags JSON,
            lang VARCHAR(10) DEFAULT 'en',
            project VARCHAR(255),
            namespace VARCHAR(255),
            is_deleted BOOLEAN NOT NULL DEFAULT 0,
            embedding_model VARCHAR(255) NOT NULL,
            PRIMARY KEY (id),
            INDEX idx_source_id (source_id),
            INDEX idx_project_namespace (project, namespace),
            INDEX idx_created_at (created_at),
            INDEX idx_is_deleted (is_deleted)
        )
        """)
        print("✅ Created new table with LONGTEXT vector column")
        
    conn.commit()
finally:
    conn.close()