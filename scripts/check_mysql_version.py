"""Check MySQL version and VECTOR support."""

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
        # Check MySQL version
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()[0]
        print(f"MySQL Version: {version}")
        
        # Check if VECTOR type is supported
        print("\nTesting VECTOR type support...")
        try:
            cursor.execute("CREATE TEMPORARY TABLE test_vector (id INT PRIMARY KEY, vec VECTOR(10))")
            print("✅ VECTOR type is supported!")
            cursor.execute("DROP TEMPORARY TABLE test_vector")
        except Exception as e:
            print(f"❌ VECTOR type is NOT supported: {e}")
            
        # Check if STRING_TO_VECTOR function exists
        print("\nTesting STRING_TO_VECTOR function...")
        try:
            cursor.execute("SELECT STRING_TO_VECTOR('[1.0, 2.0, 3.0]')")
            print("✅ STRING_TO_VECTOR function is supported!")
        except Exception as e:
            print(f"❌ STRING_TO_VECTOR function is NOT supported: {e}")
            
        # Check if DISTANCE function exists
        print("\nTesting DISTANCE function...")
        try:
            cursor.execute("SELECT DISTANCE(STRING_TO_VECTOR('[1,2,3]'), STRING_TO_VECTOR('[4,5,6]'), 'COSINE')")
            print("✅ DISTANCE function is supported!")
        except Exception as e:
            print(f"❌ DISTANCE function is NOT supported: {e}")
            
finally:
    conn.close()