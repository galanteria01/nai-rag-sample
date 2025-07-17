# RAG Application

A Retrieval-Augmented Generation application with Streamlit interface for document-based chat using Nutanix Enterprise AI.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export NUTANIX_API_KEY="your-api-key"
   export NUTANIX_ENDPOINT="https://your-cluster/v1"
   ```

## Run Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Configuration

1. Enter your Nutanix API key and endpoint in the sidebar
2. Configure embedding and chat models
3. Click "Initialize System"
4. Upload documents or use sample content
5. Start chatting with your documents

## Features

- **RAG Mode**: Chat with uploaded documents
- **Direct Chat**: Direct conversation with the language model
- **MCP Tools**: Enhanced capabilities (when RAG is disabled)
- **Database Query Tool**: Execute LLM-generated SQL queries with table results
- **Multiple file formats**: PDF, TXT, DOCX, MD, and more

## Database Query Tool

The application includes a powerful database query tool that allows the LLM to generate and execute SQL queries against your databases:

### Supported Databases
- SQLite (built-in)
- PostgreSQL (via psycopg2)
- MySQL (via pymysql)
- Any SQLAlchemy-compatible database

### Features
- **Safety-first**: Only SELECT queries allowed
- **Query validation**: Prevents dangerous operations
- **Table formatting**: Results displayed as formatted tables
- **Row limiting**: Configurable result limits
- **Multiple formats**: Markdown and HTML table output

### Usage Example

```python
# Enable database query tool
tools_manager = MCPToolsManager(enabled_tools=["database_query"])

# Execute a query
result = tools_manager.call_tool(
    "execute_database_query",
    {
        "query": "SELECT * FROM employees WHERE department = 'Engineering'",
        "database_url": "sqlite:///sample.db",
        "max_rows": 100
    }
)
```

### Demo

Run the database demo to see the tool in action:

```bash
python database_demo.py
```

This creates a sample SQLite database and demonstrates various query capabilities. 