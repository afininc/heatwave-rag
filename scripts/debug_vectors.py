"""Debug script to check vector storage and search."""

import os
from dotenv import load_dotenv
import pymysql
import numpy as np

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
        # Check table structure
        cursor.execute("DESCRIBE vector_documents")
        print("Table structure:")
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]}")
        
        print("\n" + "="*50 + "\n")
        
        # Check if any documents exist
        cursor.execute("SELECT COUNT(*) FROM vector_documents WHERE is_deleted = 0")
        count = cursor.fetchone()[0]
        print(f"Total active documents: {count}")
        
        # Check a sample document
        cursor.execute("""
            SELECT id, LENGTH(vector) as vector_length, 
                   LEFT(text, 100) as text_preview,
                   project, namespace
            FROM vector_documents 
            WHERE is_deleted = 0
            LIMIT 5
        """)
        
        print("\nSample documents:")
        for row in cursor.fetchall():
            print(f"\nID: {row[0]}")
            print(f"Vector length (bytes): {row[1]}")
            print(f"Text preview: {row[2]}...")
            print(f"Project: {row[3]}, Namespace: {row[4]}")
            
        # Try to check vector format
        cursor.execute("""
            SELECT id, HEX(LEFT(vector, 16)) as vector_hex
            FROM vector_documents 
            WHERE is_deleted = 0
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        if row:
            print(f"\nVector hex preview (first 16 bytes): {row[1]}")
            
finally:
    conn.close()