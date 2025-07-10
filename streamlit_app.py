"""
Streamlit Chat Interface for RAG Application

This module provides a web-based chat interface for the RAG application using Streamlit.
"""

import streamlit as st
import os
from typing import Dict, Any, List
import json
import tempfile
from datetime import datetime
import time

# Import RAG components
from rag_app import EmbeddingService, VectorStore, DocumentProcessor, RAGEngine


# Configure Streamlit page
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left: 4px solid #4f8bf9;
    }
    .assistant-message {
        background-color: #ffffff;
        border-left: 4px solid #00cc88;
    }
    .document-card {
        background-color: #f9f9f9;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 3px solid #ff6b6b;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.getenv('OPENAI_API_KEY', '')
    if 'base_url' not in st.session_state:
        st.session_state.base_url = os.getenv('OPENAI_BASE_URL', '')
    if 'knowledge_base_loaded' not in st.session_state:
        st.session_state.knowledge_base_loaded = False


def setup_rag_engine():
    """Setup the RAG engine with user configuration."""
    try:
        # Prepare custom headers for Azure
        custom_headers = {}
        if st.session_state.endpoint_type == "Azure OpenAI" and hasattr(st.session_state, 'api_version'):
            custom_headers["api-version"] = st.session_state.api_version
        
        # Initialize embedding service based on endpoint type
        if st.session_state.endpoint_type == "Nutanix Enterprise AI":
            embedding_service = EmbeddingService.create_for_nutanix(
                api_key=st.session_state.api_key,
                base_url=st.session_state.base_url,
                model_name=st.session_state.embedding_model
            )
            # Use predefined dimension for Nutanix if available
            embedding_dim = getattr(st.session_state, 'embedding_dimension', None)
            if not embedding_dim:
                embedding_dim = embedding_service.get_embedding_dimension()
        elif st.session_state.endpoint_type == "Azure OpenAI":
            embedding_service = EmbeddingService.create_for_azure(
                api_key=st.session_state.api_key,
                base_url=st.session_state.base_url,
                model_name=st.session_state.embedding_model,
                api_version=getattr(st.session_state, 'api_version', '2023-05-15')
            )
            embedding_dim = embedding_service.get_embedding_dimension()
        else:
            # OpenAI or Custom Endpoint
            embedding_service = EmbeddingService(
                api_key=st.session_state.api_key,
                base_url=st.session_state.base_url if st.session_state.base_url else None,
                model_name=st.session_state.embedding_model,
                custom_headers=custom_headers if custom_headers else None
            )
            embedding_dim = embedding_service.get_embedding_dimension()
        
        # Initialize vector store
        vector_store = VectorStore(
            dimension=embedding_dim,
            index_type=st.session_state.index_type
        )
        
        # Initialize document processor
        document_processor = DocumentProcessor(
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
            chunking_strategy=st.session_state.chunking_strategy
        )
        
        # Initialize RAG engine
        rag_engine = RAGEngine(
            embedding_service=embedding_service,
            vector_store=vector_store,
            document_processor=document_processor,
            api_key=st.session_state.api_key,
            base_url=st.session_state.base_url if st.session_state.base_url else None,
            model_name=st.session_state.chat_model,
            temperature=st.session_state.temperature,
            max_retrieved_docs=st.session_state.max_retrieved_docs,
            custom_headers=custom_headers if custom_headers else None
        )
        
        st.session_state.rag_engine = rag_engine
        return True
        
    except Exception as e:
        st.error(f"Error setting up RAG engine: {str(e)}")
        return False


def sidebar_configuration():
    """Create the sidebar configuration panel."""
    st.sidebar.header("⚙️ Configuration")
    
    # Endpoint Type Selection
    st.sidebar.subheader("🌐 API Endpoint")
    endpoint_type = st.sidebar.selectbox(
        "Select API Provider",
        ["OpenAI", "Nutanix Enterprise AI", "Azure OpenAI", "Custom Endpoint"],
        help="Choose your API provider"
    )
    
    # Configure based on endpoint type
    if endpoint_type == "OpenAI":
        st.session_state.api_key = st.sidebar.text_input(
            "OpenAI API Key",
            value=st.session_state.api_key,
            type="password",
            help="Enter your OpenAI API key"
        )
        st.session_state.base_url = ""
        
        # Model Configuration for OpenAI
        st.sidebar.subheader("🤖 Model Settings")
        st.session_state.embedding_model = st.sidebar.selectbox(
            "Embedding Model",
            ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            help="Choose the embedding model"
        )
        
        st.session_state.chat_model = st.sidebar.selectbox(
            "Chat Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
            help="Choose the chat model"
        )
    
    elif endpoint_type == "Nutanix Enterprise AI":
        st.session_state.api_key = st.sidebar.text_input(
            "Nutanix API Key",
            value=st.session_state.api_key,
            type="password",
            help="Enter your Nutanix Enterprise AI API key"
        )
        
        st.session_state.base_url = st.sidebar.text_input(
            "Nutanix Endpoint URL",
            value=st.session_state.base_url,
            placeholder="https://your-nutanix-endpoint/v1",
            help="Enter your Nutanix Enterprise AI endpoint URL"
        )
        
        # Model Configuration for Nutanix
        st.sidebar.subheader("🤖 Model Settings")
        st.session_state.embedding_model = st.sidebar.text_input(
            "Embedding Model Name",
            value=getattr(st.session_state, 'embedding_model', 'text-embedding-3-small'),
            help="Enter the exact embedding model name from Nutanix"
        )
        
        st.session_state.chat_model = st.sidebar.text_input(
            "Chat Model Name", 
            value=getattr(st.session_state, 'chat_model', 'gpt-4o-mini'),
            help="Enter the exact chat model name from Nutanix"
        )
        
        # Additional Nutanix-specific settings
        st.session_state.embedding_dimension = st.sidebar.number_input(
            "Embedding Dimension",
            min_value=128,
            max_value=4096,
            value=getattr(st.session_state, 'embedding_dimension', 1536),
            help="Dimension of embeddings from your model"
        )
    
    elif endpoint_type == "Azure OpenAI":
        st.session_state.api_key = st.sidebar.text_input(
            "Azure API Key",
            value=st.session_state.api_key,
            type="password",
            help="Enter your Azure OpenAI API key"
        )
        
        st.session_state.base_url = st.sidebar.text_input(
            "Azure Endpoint URL",
            value=st.session_state.base_url,
            placeholder="https://your-resource.openai.azure.com/",
            help="Enter your Azure OpenAI endpoint URL"
        )
        
        # Model Configuration for Azure
        st.sidebar.subheader("🤖 Model Settings")
        st.session_state.embedding_model = st.sidebar.text_input(
            "Embedding Deployment Name",
            value=getattr(st.session_state, 'embedding_model', 'text-embedding-3-small'),
            help="Enter your embedding model deployment name"
        )
        
        st.session_state.chat_model = st.sidebar.text_input(
            "Chat Deployment Name",
            value=getattr(st.session_state, 'chat_model', 'gpt-4o-mini'),
            help="Enter your chat model deployment name"
        )
        
        st.session_state.api_version = st.sidebar.text_input(
            "API Version",
            value=getattr(st.session_state, 'api_version', '2023-05-15'),
            help="Azure OpenAI API version"
        )
    
    else:  # Custom Endpoint
        st.session_state.api_key = st.sidebar.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            help="Enter your API key"
        )
        
        st.session_state.base_url = st.sidebar.text_input(
            "Custom Endpoint URL",
            value=st.session_state.base_url,
            placeholder="https://your-custom-endpoint/v1",
            help="Enter your custom API endpoint URL"
        )
        
        # Model Configuration for Custom
        st.sidebar.subheader("🤖 Model Settings")
        st.session_state.embedding_model = st.sidebar.text_input(
            "Embedding Model Name",
            value=getattr(st.session_state, 'embedding_model', 'text-embedding-3-small'),
            help="Enter the embedding model name"
        )
        
        st.session_state.chat_model = st.sidebar.text_input(
            "Chat Model Name",
            value=getattr(st.session_state, 'chat_model', 'gpt-4o-mini'),
            help="Enter the chat model name"
        )
    
    # Store endpoint type for later use
    st.session_state.endpoint_type = endpoint_type
    
    st.session_state.temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness in responses"
    )
    
    # RAG Configuration
    st.sidebar.subheader("RAG Settings")
    st.session_state.chunk_size = st.sidebar.slider(
        "Chunk Size",
        min_value=200,
        max_value=2000,
        value=1000,
        step=100,
        help="Size of text chunks for processing"
    )
    
    st.session_state.chunk_overlap = st.sidebar.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        help="Overlap between consecutive chunks"
    )
    
    st.session_state.chunking_strategy = st.sidebar.selectbox(
        "Chunking Strategy",
        ["recursive", "token"],
        help="Strategy for splitting documents"
    )
    
    st.session_state.index_type = st.sidebar.selectbox(
        "Vector Index Type",
        ["flat", "ivf", "hnsw"],
        help="Type of vector index for similarity search"
    )
    
    st.session_state.max_retrieved_docs = st.sidebar.slider(
        "Max Retrieved Documents",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Maximum number of documents to retrieve for context"
    )
    
    # Connection Testing
    st.sidebar.subheader("🔗 Connection Testing")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Test Connection", help="Test API connection before initializing"):
            if st.session_state.api_key and (not st.session_state.base_url or st.session_state.endpoint_type == "OpenAI"):
                with st.spinner("Testing connection..."):
                    try:
                        # Create a temporary embedding service to test
                        if st.session_state.endpoint_type == "Azure OpenAI":
                            test_service = EmbeddingService.create_for_azure(
                                api_key=st.session_state.api_key,
                                base_url=st.session_state.base_url,
                                model_name=st.session_state.embedding_model,
                                api_version=getattr(st.session_state, 'api_version', '2023-05-15')
                            )
                        elif st.session_state.endpoint_type == "Nutanix Enterprise AI":
                            test_service = EmbeddingService.create_for_nutanix(
                                api_key=st.session_state.api_key,
                                base_url=st.session_state.base_url,
                                model_name=st.session_state.embedding_model
                            )
                        else:
                            test_service = EmbeddingService(
                                api_key=st.session_state.api_key,
                                base_url=st.session_state.base_url if st.session_state.base_url else None,
                                model_name=st.session_state.embedding_model
                            )
                        
                        result = test_service.test_connection()
                        if result["success"]:
                            st.sidebar.success(f"✅ Connection successful!\nEndpoint: {result.get('endpoint_type', 'Unknown')}\nModel: {result['model']}\nDimension: {result['embedding_dimension']}")
                        else:
                            st.sidebar.error(f"❌ Connection failed: {result['error']}")
                    except Exception as e:
                        st.sidebar.error(f"❌ Connection test failed: {str(e)}")
            else:
                st.sidebar.warning("Please configure API key and endpoint URL")
    
    with col2:
        # Initialize/Reset RAG Engine
        if st.button("Initialize RAG", type="primary", help="Initialize the RAG engine"):
            if st.session_state.api_key:
                with st.spinner("Initializing RAG engine..."):
                    if setup_rag_engine():
                        # Test the complete RAG engine
                        connection_test = st.session_state.rag_engine.test_connection()
                        if connection_test["overall_success"]:
                            st.sidebar.success("🎉 RAG engine initialized successfully!")
                            
                            # Show configuration summary
                            stats = st.session_state.rag_engine.get_stats()
                            st.sidebar.info(f"""
                            **Configuration Summary:**
                            - Endpoint: {stats['endpoint_type']}
                            - Chat Model: {stats['model_name']}
                            - Embedding Model: {stats['embedding_service_info']['model_name']}
                            - Vector Dimension: {stats['embedding_service_info']['embedding_dimension']}
                            """)
                            
                            st.session_state.knowledge_base_loaded = False
                        else:
                            st.sidebar.error("⚠️ RAG engine initialized but connection test failed")
            else:
                st.sidebar.error("Please enter your API key")


def document_management():
    """Create the document management interface."""
    st.sidebar.header("📚 Knowledge Base")
    
    if st.session_state.rag_engine is None:
        st.sidebar.warning("Please initialize the RAG engine first")
        return
    
    # File Upload
    st.sidebar.subheader("Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx', 'md', 'py', 'js', 'html', 'json', 'csv'],
        help="Upload documents to add to the knowledge base"
    )
    
    if uploaded_files:
        if st.sidebar.button("Process Uploaded Files"):
            with st.spinner("Processing documents..."):
                processed_count = 0
                for uploaded_file in uploaded_files:
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Process the file
                        doc_ids = st.session_state.rag_engine.add_document_from_file(
                            tmp_file_path,
                            metadata={"uploaded_by": "user", "original_name": uploaded_file.name}
                        )
                        processed_count += len(doc_ids)
                        
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
                        
                    except Exception as e:
                        st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                if processed_count > 0:
                    st.sidebar.success(f"Successfully processed {processed_count} document chunks!")
                    st.session_state.knowledge_base_loaded = True
    
    # Text Input
    st.sidebar.subheader("Add Text")
    text_input = st.sidebar.text_area(
        "Enter text to add to knowledge base",
        height=100,
        help="Enter raw text to add to the knowledge base"
    )
    
    if text_input and st.sidebar.button("Add Text"):
        with st.spinner("Processing text..."):
            try:
                doc_ids = st.session_state.rag_engine.add_text(
                    text_input,
                    source="user_input",
                    metadata={"added_by": "user", "timestamp": datetime.now().isoformat()}
                )
                st.sidebar.success(f"Successfully added {len(doc_ids)} text chunks!")
                st.session_state.knowledge_base_loaded = True
            except Exception as e:
                st.sidebar.error(f"Error adding text: {str(e)}")
    
    # Knowledge Base Stats
    if st.session_state.rag_engine and st.session_state.knowledge_base_loaded:
        st.sidebar.subheader("Knowledge Base Stats")
        stats = st.session_state.rag_engine.get_stats()
        vector_stats = stats.get("vector_store_stats", {})
        
        st.sidebar.metric("Total Documents", vector_stats.get("total_documents", 0))
        st.sidebar.metric("Vector Dimension", vector_stats.get("dimension", 0))
        st.sidebar.metric("Index Type", vector_stats.get("index_type", "unknown"))
    
    # Save/Load Knowledge Base
    st.sidebar.subheader("Save/Load Knowledge Base")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Save KB"):
            if st.session_state.rag_engine:
                try:
                    kb_path = f"knowledge_base_{int(time.time())}"
                    st.session_state.rag_engine.save_knowledge_base(kb_path)
                    st.sidebar.success(f"Knowledge base saved as {kb_path}")
                except Exception as e:
                    st.sidebar.error(f"Error saving: {str(e)}")
    
    with col2:
        kb_file = st.file_uploader("Load KB", type=['metadata'])
        if kb_file:
            # This would need additional implementation for file handling
            st.sidebar.info("Knowledge base loading from file not fully implemented")


def chat_interface():
    """Create the main chat interface."""
    st.title("🤖 RAG Chat Assistant")
    
    if st.session_state.rag_engine is None:
        st.warning("Please configure and initialize the RAG engine in the sidebar.")
        return
    
    # Chat History Display
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show retrieved documents if available
                    if "retrieved_docs" in message:
                        with st.expander(f"📄 Retrieved Documents ({len(message['retrieved_docs'])} docs)"):
                            for j, doc_info in enumerate(message["retrieved_docs"]):
                                st.markdown(f"""
                                <div class="document-card">
                                    <strong>Document {j+1}</strong> (Score: {doc_info['similarity_score']:.3f})<br>
                                    <small>Source: {doc_info['metadata'].get('source', 'Unknown')}</small><br>
                                    <em>{doc_info['content'][:200]}...</em>
                                </div>
                                """, unsafe_allow_html=True)
        else:
            st.info("👋 Welcome! Start by asking a question or uploading some documents to create a knowledge base.")
    
    # Chat Input
    st.markdown("---")
    user_input = st.text_input(
        "Ask a question:",
        placeholder="Type your question here...",
        key="user_input"
    )
    
    # Chat Options
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        include_context = st.checkbox("Use Context", value=True, help="Include retrieved documents in the response")
    
    with col2:
        use_history = st.checkbox("Use History", value=True, help="Include chat history for context")
    
    with col3:
        if st.button("Clear Chat", help="Clear chat history"):
            st.session_state.chat_history = []
            if st.session_state.rag_engine:
                st.session_state.rag_engine.clear_chat_history()
            st.rerun()
    
    # Process User Input
    if user_input:
        with st.spinner("Thinking..."):
            try:
                # Get response from RAG engine
                response_data = st.session_state.rag_engine.chat(
                    user_input,
                    include_context=include_context,
                    use_chat_history=use_history
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                assistant_message = {
                    "role": "assistant",
                    "content": response_data["response"]
                }
                
                # Include retrieved documents if available
                if response_data.get("retrieved_documents"):
                    assistant_message["retrieved_docs"] = response_data["retrieved_documents"]
                
                st.session_state.chat_history.append(assistant_message)
                
                # Rerun to show the new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")


def main():
    """Main application function."""
    initialize_session_state()
    
    # Create sidebar configuration
    sidebar_configuration()
    
    # Create document management interface
    document_management()
    
    # Create main chat interface
    chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit | RAG Application with OpenAI-compatible APIs",
        help="This application uses Retrieval-Augmented Generation (RAG) to answer questions based on your documents."
    )


if __name__ == "__main__":
    main() 