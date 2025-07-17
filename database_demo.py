#!/usr/bin/env python3
"""
Database Query Tool Demo

This script demonstrates how to use the database query tool in the RAG application.
It creates a sample SQLite database with test data and shows how to query it.
"""

import sqlite3
import os
from rag_app.mcp_tools import MCPToolsManager

def create_sample_database():
    """Create a sample SQLite database with test data."""
    db_path = "sample_data.sqlite"
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create connection and cursor
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary REAL NOT NULL,
            hire_date TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            manager TEXT NOT NULL,
            budget REAL NOT NULL
        )
    """)
    
    # Insert sample data
    employees_data = [
        (1, 'Alice Johnson', 'Engineering', 95000.0, '2022-01-15'),
        (2, 'Bob Smith', 'Marketing', 72000.0, '2021-06-20'),
        (3, 'Carol Davis', 'Engineering', 88000.0, '2023-03-10'),
        (4, 'David Wilson', 'Sales', 65000.0, '2022-08-05'),
        (5, 'Eva Brown', 'HR', 78000.0, '2021-11-12'),
        (6, 'Frank Miller', 'Engineering', 102000.0, '2020-09-18'),
        (7, 'Grace Lee', 'Marketing', 69000.0, '2023-01-25'),
        (8, 'Henry Taylor', 'Sales', 71000.0, '2022-04-14')
    ]
    
    departments_data = [
        (1, 'Engineering', 'Frank Miller', 500000.0),
        (2, 'Marketing', 'Bob Smith', 200000.0),
        (3, 'Sales', 'David Wilson', 300000.0),
        (4, 'HR', 'Eva Brown', 150000.0)
    ]
    
    cursor.executemany(
        "INSERT INTO employees (id, name, department, salary, hire_date) VALUES (?, ?, ?, ?, ?)",
        employees_data
    )
    
    cursor.executemany(
        "INSERT INTO departments (id, name, manager, budget) VALUES (?, ?, ?, ?)",
        departments_data
    )
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"✅ Sample database created: {db_path}")
    return db_path

def demo_database_queries():
    """Demonstrate various database queries using the MCP tools."""
    
    # Create sample database
    db_path = create_sample_database()
    database_url = f"sqlite:///{db_path}"
    
    # Initialize MCP tools manager with database query tool enabled
    tools_manager = MCPToolsManager(enabled_tools=["database_query"])
    
    # Sample queries to demonstrate
    sample_queries = [
        {
            "description": "Get all employees",
            "query": "SELECT * FROM employees"
        },
        {
            "description": "Get employees by department",
            "query": "SELECT name, salary FROM employees WHERE department = 'Engineering' ORDER BY salary DESC"
        },
        {
            "description": "Get department statistics",
            "query": "SELECT department, COUNT(*) as employee_count, AVG(salary) as avg_salary FROM employees GROUP BY department"
        },
        {
            "description": "Get employees hired after 2022",
            "query": "SELECT name, department, hire_date FROM employees WHERE hire_date > '2022-01-01' ORDER BY hire_date"
        },
        {
            "description": "Join employees with department budgets",
            "query": """
                SELECT e.name, e.department, e.salary, d.budget 
                FROM employees e 
                JOIN departments d ON e.department = d.name 
                ORDER BY d.budget DESC, e.salary DESC
            """
        }
    ]
    
    print("\n🗃️ Database Query Tool Demo")
    print("=" * 50)
    
    for i, query_info in enumerate(sample_queries, 1):
        print(f"\n{i}. {query_info['description']}")
        print("-" * 30)
        
        # Execute query using the tool
        result = tools_manager.call_tool(
            "execute_database_query",
            {
                "query": query_info["query"],
                "database_url": database_url,
                "max_rows": 50
            }
        )
        
        if result.success:
            print("✅ Query executed successfully!")
            # Format and display result
            formatted_result = tools_manager._format_database_query_result(result.result)
            print(formatted_result)
        else:
            print(f"❌ Query failed: {result.error}")
    
    print(f"\n🧹 Cleaning up: removing {db_path}")
    os.remove(db_path)

def demo_llm_generated_query():
    """Demonstrate how an LLM might generate and execute a query."""
    
    # Create sample database
    db_path = create_sample_database()
    database_url = f"sqlite:///{db_path}"
    
    # Initialize tools manager
    tools_manager = MCPToolsManager(enabled_tools=["database_query"])
    
    print("\n🤖 LLM-Generated Query Demo")
    print("=" * 40)
    
    # Simulate an LLM-generated query based on a user question
    user_question = "Show me the highest paid employees in each department"
    
    # This is what an LLM might generate
    llm_generated_query = """
        SELECT 
            department,
            name,
            salary,
            RANK() OVER (PARTITION BY department ORDER BY salary DESC) as salary_rank
        FROM employees
        WHERE salary = (
            SELECT MAX(salary) 
            FROM employees e2 
            WHERE e2.department = employees.department
        )
        ORDER BY department, salary DESC
    """
    
    print(f"User Question: '{user_question}'")
    print(f"LLM Generated Query: {llm_generated_query.strip()}")
    print()
    
    # Execute the query
    result = tools_manager.call_tool(
        "execute_database_query",
        {
            "query": llm_generated_query,
            "database_url": database_url,
            "max_rows": 20
        }
    )
    
    if result.success:
        print("✅ LLM query executed successfully!")
        formatted_result = tools_manager._format_database_query_result(result.result)
        print(formatted_result)
    else:
        print(f"❌ LLM query failed: {result.error}")
    
    # Clean up
    os.remove(db_path)

if __name__ == "__main__":
    print("🚀 Database Query Tool Demonstration")
    print("This demo shows how to use the database query tool with sample data.")
    
    try:
        demo_database_queries()
        demo_llm_generated_query()
        
        print("\n✅ Demo completed successfully!")
        print("\nTo use in your RAG application:")
        print("1. Enable 'database_query' in MCPToolsManager")
        print("2. Provide your database connection URL")
        print("3. Let the LLM generate SQL queries based on user questions")
        print("4. Results will be displayed as formatted tables in the chat")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc() 