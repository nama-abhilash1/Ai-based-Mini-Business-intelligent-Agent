import os
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_sql_query_chain

def get_db_connection():
    # Read-only mode to prevent destructive operations
    return sqlite3.connect('file:sales.db?mode=ro', uri=True)

def validate_query(query: str) -> bool:
    """Ensure the query is a SELECT statement and has no destructive operations."""
    query_upper = query.upper()
    if not query_upper.strip().startswith("SELECT"):
        return False
    destructive_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "GRANT", "REVOKE"]
    for kw in destructive_keywords:
        if kw in query_upper:
            return False
    return True

def query_database(question: str, api_key: str):
    if not os.path.exists("sales.db"):
        return "Database not found. Please run setup_db.py first.", None, None

    # Connect DB to Langchain
    db = SQLDatabase.from_uri("sqlite:///sales.db")
    
    # Use Google Gemini as the LLM (or customize this to OpenAI)
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, google_api_key=api_key)
        chain = create_sql_query_chain(llm, db)
        
        # 1. Generate SQL
        response = chain.invoke({"question": question})
        sql_query = response.replace("```sql", "").replace("```", "").strip()
        
        # 2. Validate SQL - block unsupported/destructive operations
        if not validate_query(sql_query):
            return "Error: Invalid or destructive SQL query generated.", sql_query, None
            
        # 3. Execute SQL safely
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        
        return "Success", sql_query, (columns, results)
        
    except Exception as e:
        return f"Error processing query: {str(e)}", None, None
