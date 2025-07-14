# RAG Application 🤖

**A production-ready Retrieval-Augmented Generation (RAG) system for enterprise document search and conversational AI.**

Transform your documents into an intelligent knowledge base that you can chat with using natural language. Built for enterprise environments with support for multiple AI providers including OpenAI, Nutanix Enterprise AI, and other OpenAI-compatible services.

## 🎯 Project Overview

### What This Project Does
This RAG application enables organizations to:
- **Upload documents** in various formats (PDF, DOCX, TXT, MD, HTML, JSON, CSV, code files)
- **Create searchable knowledge bases** using advanced vector embeddings
- **Chat with documents** using natural language queries
- **Deploy enterprise-grade AI solutions** with multiple provider support

### Who This Is For
- **Enterprise teams** looking to implement document-based AI solutions
- **Developers** building custom RAG applications
- **Organizations using Nutanix Enterprise AI** (dedicated interface included)
- **Anyone** wanting to chat with their document collections

### Key Value Propositions
- 🚀 **Quick deployment** - Get running in minutes, not hours
- 🔧 **Multi-provider support** - Works with OpenAI, Nutanix, Azure, and custom endpoints
- 📊 **Enterprise ready** - Built for scale with proper error handling and logging
- 🎨 **User-friendly** - Web interface requires no technical knowledge
- 🔒 **Secure** - API keys and sensitive data handled properly

## ✨ Core Features

| Feature | Description |
|---------|-------------|
| **Multi-format Support** | PDF, DOCX, TXT, MD, HTML, JSON, CSV, Python, JavaScript |
| **Smart Chunking** | Intelligent text splitting with configurable overlap |
| **Vector Search** | FAISS-powered similarity search with multiple index types |
| **Conversation Memory** | Maintains context across chat sessions |
| **Metadata Filtering** | Advanced document filtering and categorization |
| **Streaming Responses** | Real-time response generation |
| **Persistence** | Save and load knowledge bases |
| **MCP Tools Integration** | Enhanced chat with web search, file ops, runtime logs, memory |
| **Enterprise Integration** | RESTful APIs for custom integrations |

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │  Jupyter Demo   │    │  Nutanix UI     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   RAG Engine    │
                    └─────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│Document Processor│ │Embedding Service│ │  Vector Store   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Core Components

- **RAG Engine**: Orchestrates retrieval and generation workflow
- **Document Processor**: Handles multiple file formats with intelligent chunking
- **Embedding Service**: Converts text to vectors using various AI providers
- **Vector Store**: Manages document storage and similarity search using FAISS

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- API key for your chosen AI provider (OpenAI, Nutanix, etc.)
- 4GB+ RAM (8GB recommended for large document sets)

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd rag-app
python setup.py  # Automated setup with virtual environment

# Or manual setup
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

1. **Edit `.env` file** (created by setup):
```bash
OPENAI_API_KEY=your_api_key_here
# Optional: Custom endpoint
OPENAI_BASE_URL=https://your-custom-endpoint/v1
```

2. **Launch the application**:
```bash
# Web interface (recommended)
streamlit run streamlit_app.py

# Nutanix Enterprise AI interface
streamlit run nutanix_example.py

# Jupyter notebook
jupyter lab  # Open rag_demo.ipynb
```

## 📚 Usage Examples

### Basic Document Chat
```python
from rag_app import RAGEngine

# Initialize with OpenAI
rag = RAGEngine.create_openai(api_key="your-key")

# Add documents
rag.add_document_from_file("company_handbook.pdf")
rag.add_documents_from_directory("policies/")

# Start chatting
answer = rag.ask("What is our vacation policy?")
```

### Enterprise Setup (Nutanix)
```python
# Quick Nutanix setup
rag = RAGEngine.create_for_nutanix(
    api_key="your-nutanix-key",
    base_url="https://your-cluster.com/v1",
    embedding_model="text-embedding-3-small",
    chat_model="llama2-7b-chat"
)

# Add enterprise documents
rag.add_documents_from_directory("company_docs/")
answer = rag.ask("How do we handle data privacy?")
```

### Advanced Configuration
```python
# Custom chunking and search
rag = RAGEngine(
    embedding_service=embedding_service,
    vector_store=vector_store,
    chunk_size=1500,
    chunk_overlap=300,
    max_retrieved_docs=10,
    temperature=0.3
)

# Metadata filtering
results = rag.search_documents(
    "security protocols",
    filter_metadata={"department": "IT", "classification": "internal"}
)
```

## 🌐 Supported AI Providers

### OpenAI (Default)
```python
rag = RAGEngine.create_openai(api_key="sk-...")
```

### Nutanix Enterprise AI
```python
rag = RAGEngine.create_for_nutanix(
    api_key="your-nutanix-key",
    base_url="https://your-cluster.com/v1",
    embedding_model="text-embedding-3-small",
    chat_model="llama2-7b-chat"
)
```

### Azure OpenAI
```python
rag = RAGEngine.create_for_azure(
    api_key="your-azure-key",
    base_url="https://your-resource.openai.azure.com/",
    embedding_model="text-embedding-3-small",
    chat_model="gpt-35-turbo"
)
```

### Custom/Local Endpoints
```python
rag = RAGEngine.create_custom(
    api_key="not-needed",
    base_url="http://localhost:11434/v1",
    embedding_model="nomic-embed-text",
    chat_model="llama2:7b"
)
```

## 🔧 MCP Tools Integration

When RAG is disabled, the system can be enhanced with MCP (Model Context Protocol) tools for advanced chat capabilities:

### Available Tools
- **🌐 Web Search**: Real-time web information lookup
- **📊 Runtime Logs**: Application monitoring and debugging
- **🐛 Runtime Errors**: Error tracking and analysis
- **📁 File Operations**: Read and write files safely
- **💻 Code Execution**: Execute code snippets (placeholder)
- **🧠 Memory Management**: Save and retrieve information

### Usage in Streamlit App
1. Disable RAG mode in the sidebar
2. Enable MCP Tools in the sidebar
3. Select desired tools
4. Chat with enhanced AI capabilities

### Programmatic Usage
```python
# Create RAG engine with MCP tools
rag = RAGEngine.create_for_nutanix(
    api_key="your-key",
    base_url="https://your-cluster.com/v1",
    embedding_model="embedcpu",
    chat_model="llama-3-3-70b",
    embedding_dimension=384,
    enable_mcp_tools=True,
    mcp_tools_config={"enabled_tools": ["web_search", "file_operations"]}
)

# Chat with tools (when RAG is disabled)
response = rag.chat_with_tools("What's the latest news about AI?")
```

## 🎛️ Configuration Options

### Document Processing
- **Chunk Size**: 500-2000 characters (default: 1000)
- **Overlap**: 0-500 characters (default: 200)
- **Strategies**: Recursive, token-based
- **File Types**: PDF, DOCX, TXT, MD, HTML, JSON, CSV, code files

### Vector Search
- **Index Types**: Flat (exact), IVF (fast), HNSW (scalable)
- **Similarity Metrics**: Cosine, Euclidean, dot product
- **Retrieval**: 1-20 documents (default: 5)

### Generation
- **Temperature**: 0.0-2.0 (default: 0.7)
- **Max Tokens**: Configurable per model
- **System Prompts**: Customizable for domain-specific responses

## 🧪 Testing & Validation

### Run Tests
```bash
python test_rag.py
```

### Test Coverage
- ✅ Component initialization
- ✅ Document processing
- ✅ Vector operations
- ✅ RAG workflow
- ✅ API integrations
- ✅ Error handling

### Nutanix Quick Test
```bash
# Set environment variables
export NUTANIX_API_KEY="your-key"
export NUTANIX_ENDPOINT="https://your-cluster.com/v1"

# Run Nutanix-specific tests
python nutanix_example.py
```

## 📊 Performance Expectations

### Document Processing
- **Small files** (< 1MB): < 10 seconds
- **Medium files** (1-10MB): 30-60 seconds
- **Large files** (10-100MB): 2-5 minutes
- **Batch processing**: Parallel processing available

### Query Response Time
- **Simple queries**: < 2 seconds
- **Complex queries**: 3-5 seconds
- **Large knowledge bases**: 5-10 seconds

### Resource Usage
- **Memory**: ~1GB base + 100MB per 1000 documents
- **Storage**: ~10MB per 1000 document chunks
- **CPU**: Moderate during processing, low during queries

## 🚨 Known Limitations

### Current Limitations
- **File size limit**: 100MB per file (can be increased)
- **Concurrent users**: Single-user Streamlit interface
- **Vector store**: In-memory (persistence requires manual save/load)
- **Language support**: Optimized for English text

### Planned Improvements
- [ ] Multi-user support with authentication
- [ ] Persistent database backends (PostgreSQL, Redis)
- [ ] Advanced metadata search and filtering
- [ ] Multi-language support
- [ ] Real-time document updates

## 🔧 Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **API Key Error** | 401 Unauthorized | Verify API key validity and permissions |
| **Model Not Found** | 404 Error | Check exact model names in your deployment |
| **Memory Issues** | Out of memory | Reduce chunk size or use IVF indexing |
| **Slow Performance** | Long response times | Use smaller embedding models or enable caching |
| **Import Errors** | Module not found | Ensure all dependencies installed |

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test connections
rag.test_connection()
```

### Getting Help
1. Check the troubleshooting section above
2. Review the [Issues](link-to-issues) page
3. Enable debug logging and review error messages
4. Contact your AI provider for model-specific issues

## 📈 Scaling Considerations

### Small Scale (< 1,000 documents)
- Use default flat indexing
- Single instance deployment
- Standard chunk sizes

### Medium Scale (1,000-10,000 documents)
- Enable IVF indexing
- Consider document metadata optimization
- Monitor memory usage

### Large Scale (10,000+ documents)
- Use HNSW indexing
- Implement document archiving
- Consider distributed deployment

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -e .
pip install pytest black isort flake8

# Run tests
pytest

# Code formatting
black .
isort .
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Deployment Guide](docs/deployment.md)

### Community
- [GitHub Issues](link-to-issues)
- [Discussions](link-to-discussions)
- [Wiki](link-to-wiki)

### Enterprise Support
For enterprise support, custom integrations, or consulting services, please contact [support@yourcompany.com](mailto:support@yourcompany.com).

---

**Built with ❤️ for the enterprise AI community**

*Ready to transform your documents into intelligent conversations? Get started in minutes!* 