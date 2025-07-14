#!/usr/bin/env python3
"""
Nutanix Enterprise AI RAG Chat Interface

This script provides a dedicated Streamlit chat interface for Nutanix Enterprise AI
with configuration in the sidebar and interactive chat in the main area.
"""

import streamlit as st
import os
import tempfile
from datetime import datetime
from rag_app import RAGEngine

# Configure Streamlit page
st.set_page_config(
    page_title="Nutanix Enterprise AI Chat",
    page_icon="🌟",
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
    .nutanix-header {
        background: linear-gradient(135deg, #0080ff 0%, #00ccff 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .rag-toggle {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #0080ff;
    }
    /* Improve chat message appearance */
    .stChatMessage {
        margin-bottom: 1rem;
    }
    /* Customize chat input */
    .stChatInput {
        position: sticky;
        bottom: 0;
        padding-top: 1rem;
        margin-top: 1rem;
    }
    /* Chat container styling */
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 0.5rem;
        background: #fafafa;
        margin-bottom: 1rem;
    }
    /* Chat header styling */
    .chat-header {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #0080ff;
    }
    /* Empty state styling */
    .empty-chat {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    /* Expander styling for knowledge base */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 0.5rem;
        border-left: 4px solid #0080ff;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'rag_enabled' not in st.session_state:
        st.session_state.rag_enabled = True
    if 'knowledge_base_loaded' not in st.session_state:
        st.session_state.knowledge_base_loaded = False
    if 'mcp_tools_enabled' not in st.session_state:
        st.session_state.mcp_tools_enabled = False

def sidebar_configuration():
    """Create the sidebar configuration panel."""
    st.sidebar.markdown("""
    <div class="nutanix-header">
        <h2>🌟 DeNuMo</h2>
        <p>Configuration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Authentication
    st.sidebar.subheader("🔐 Authentication")
    nutanix_api_key = st.sidebar.text_input(
        "API Key",
        value=os.getenv("NUTANIX_API_KEY", ""),
        type="password",
        help="Enter your Nutanix Enterprise AI API key",
        placeholder="your-nutanix-api-key"
    )
    
    # Endpoint Configuration
    st.sidebar.subheader("🌐 Endpoint")
    nutanix_endpoint = st.sidebar.text_input(
        "Endpoint URL",
        value=os.getenv("NUTANIX_ENDPOINT", ""),
        help="Enter your Nutanix Enterprise AI endpoint URL (must end with /v1)",
        placeholder="https://your-nutanix-cluster/v1"
    )
    
    # Endpoint validation
    if nutanix_endpoint and not nutanix_endpoint.endswith('/v1'):
        st.sidebar.warning("⚠️ Endpoint URL should end with '/v1'")
    
    # Model Configuration
    st.sidebar.subheader("🤖 Models")
    st.sidebar.info("💡 Use exact model names from your Nutanix deployment")
    
    # Show common model examples
    with st.sidebar.expander("📋 Common Model Names"):
        st.markdown("""
        **Common Embedding Models:**
        - `text-embedding-3-small`
        - `text-embedding-3-large` 
        - `text-embedding-ada-002`
        - `sentence-transformers/all-MiniLM-L6-v2`
        
        **Common Chat Models:**
        - `llama2-7b-chat`
        - `llama2-13b-chat`
        - `mixtral-8x7b-instruct`
        - `gpt-3.5-turbo`
        - `gpt-4`
        
        ⚠️ **Note**: Use exact names from your deployment
        """)
    
    embedding_model = st.sidebar.text_input(
        "Embedding Model",
        value="embedcpu",
        help="Enter the exact embedding model name available in your Nutanix deployment"
        
    )
    
    chat_model = st.sidebar.text_input(
        "Chat Model",
        value="llama-3-3-70b",
        help="Enter the exact chat model name available in your Nutanix deployment"
    )
    
    embedding_dimension = st.sidebar.number_input(
        "Embedding Dimension",
        min_value=128,
        max_value=4096,
        value=384,
        help="Dimension of embeddings from your model"
    )
    
    # Advanced Settings
    st.sidebar.subheader("⚙️ Advanced Settings")
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness in responses"
    )
    
    max_retrieved_docs = st.sidebar.slider(
        "Max Retrieved Documents",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Maximum number of documents to retrieve for context"
    )
    
    # RAG Settings (only shown if RAG is enabled)
    if st.session_state.rag_enabled:
        st.sidebar.subheader("📄 RAG Settings")
        
        chunk_size = st.sidebar.slider(
            "Chunk Size",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of text chunks for processing"
        )
        
        chunk_overlap = st.sidebar.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Overlap between consecutive chunks"
        )
        
        chunking_strategy = st.sidebar.selectbox(
            "Chunking Strategy",
            ["recursive", "token"],
            help="Strategy for splitting documents"
        )
        
        index_type = st.sidebar.selectbox(
            "Vector Index Type",
            ["flat", "ivf", "hnsw"],
            help="Type of vector index for similarity search"
        )
        
        # Default MCP tools settings when RAG is enabled
        enable_mcp_tools = False
        mcp_tools_config = {}
    else:
        # Default values when RAG is disabled
        chunk_size = 1000
        chunk_overlap = 200
        chunking_strategy = "recursive"
        index_type = "flat"
        
        # MCP Tools Settings (only shown when RAG is disabled)
        st.sidebar.subheader("🔧 MCP Tools Settings")
        
        enable_mcp_tools = st.sidebar.checkbox(
            "Enable MCP Tools",
            value=True,
            help="Enable MCP tools for enhanced chat capabilities"
        )
        
        if enable_mcp_tools:
            st.sidebar.write("**Available Tools:**")
            
            # Tool selection
            available_tools = {
                # "web_search": "🌐 Web Search",
                # "runtime_logs": "📊 Runtime Logs",
                # "runtime_errors": "🐛 Runtime Errors",
                "file_operations": "📁 File Operations",
                "code_execution": "💻 Code Execution",
                "memory_management": "🧠 Memory Management"
            }
            
            selected_tools = []
            for tool_id, tool_name in available_tools.items():
                if st.sidebar.checkbox(tool_name, value=True, key=f"tool_{tool_id}"):
                    selected_tools.append(tool_id)
            
            mcp_tools_config = {
                "enabled_tools": selected_tools
            }
            
            if selected_tools:
                st.sidebar.success(f"✅ {len(selected_tools)} tools enabled")
            else:
                st.sidebar.warning("⚠️ No tools selected")
        else:
            mcp_tools_config = {}
    
    # Connection and Initialization
    st.sidebar.subheader("🔗 Connection")
    
    # Configuration Status
    config_complete = nutanix_api_key and nutanix_endpoint and embedding_model and chat_model
    
    if config_complete:
        st.sidebar.success("✅ Configuration Complete")
    else:
        st.sidebar.warning("⚠️ Please complete configuration")
    
    # Test Connection Button (separate from initialization)
    if config_complete and st.sidebar.button("🔍 Test Connection"):
        with st.spinner("Testing connection to Nutanix Enterprise AI..."):
            try:
                # Test with a simple embedding request first
                from rag_app import EmbeddingService
                
                test_embedding_service = EmbeddingService.create_for_nutanix(
                    api_key=nutanix_api_key,
                    base_url=nutanix_endpoint,
                    model_name=embedding_model
                )
                
                # Test embedding service
                embedding_test = test_embedding_service.test_connection()
                
                if embedding_test["success"]:
                    st.sidebar.success("✅ Embedding service connected!")
                    st.sidebar.write(f"Model: {embedding_test['model']}")
                    st.sidebar.write(f"Dimension: {embedding_test['embedding_dimension']}")
                else:
                    st.sidebar.error("❌ Embedding service failed!")
                    st.sidebar.error(f"Error: {embedding_test.get('error', 'Unknown error')}")
                    
                    # Provide specific troubleshooting for 404 errors
                    error_msg = str(embedding_test.get('error', '')).lower()
                    if '404' in error_msg or 'not found' in error_msg:
                        st.sidebar.error("""
                        **Troubleshooting 404 Error:**
                        1. Check that your endpoint URL is correct
                        2. Verify the embedding model name exists in your deployment
                        3. Ensure the endpoint URL ends with '/v1'
                        4. Contact your Nutanix administrator to verify model availability
                        """)
                
            except Exception as e:
                st.sidebar.error(f"❌ Connection test failed: {str(e)}")
                
                # Provide detailed error information
                error_msg = str(e).lower()
                if '404' in error_msg or 'not found' in error_msg:
                    st.sidebar.error("""
                    **404 Error Troubleshooting:**
                    1. **Check Endpoint URL**: Ensure it ends with '/v1'
                    2. **Verify Model Names**: Use exact names from your Nutanix deployment
                    3. **Check Network Access**: Ensure you can reach the Nutanix cluster
                    4. **Contact Admin**: Verify model deployment status
                    """)
    
    # Initialize RAG Engine
    if config_complete and st.sidebar.button("🚀 Initialize System", type="primary"):
        with st.spinner("Initializing Nutanix Enterprise AI system..."):
            try:
                # Create RAG engine
                rag_engine = RAGEngine.create_for_nutanix(
                    api_key=nutanix_api_key,
                    base_url=nutanix_endpoint,
                    embedding_model=embedding_model,
                    chat_model=chat_model,
                    embedding_dimension=embedding_dimension,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chunking_strategy=chunking_strategy,
                    temperature=temperature,
                    max_retrieved_docs=max_retrieved_docs,
                    index_type=index_type,
                    enable_mcp_tools=enable_mcp_tools,
                    mcp_tools_config=mcp_tools_config
                )
                
                st.session_state.rag_engine = rag_engine
                st.sidebar.success("✅ System initialized successfully!")
                
                # Test connection
                connection_test = rag_engine.test_connection()
                if connection_test["overall_success"]:
                    st.sidebar.success("✅ Connection verified!")
                    
                    # Show detailed connection info
                    st.sidebar.write("**Connection Details:**")
                    st.sidebar.write(f"- Embedding: {connection_test['embedding_service']['model']}")
                    st.sidebar.write(f"- Chat: {connection_test['chat_service']['model']}")
                else:
                    st.sidebar.error("❌ Connection test failed!")
                    st.sidebar.error(f"Embedding: {connection_test['embedding_service']['message']}")
                    st.sidebar.error(f"Chat: {connection_test['chat_service']['message']}")
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
                st.session_state.rag_engine = None
                
                # Enhanced error handling for 404
                error_msg = str(e).lower()
                if '404' in error_msg or 'not found' in error_msg:
                    st.sidebar.error("""
                    **Possible Solutions:**
                    1. Check model names in your Nutanix deployment
                    2. Verify endpoint URL format
                    3. Ensure models are deployed and running
                    4. Contact your Nutanix administrator
                    """)
    
    # System Status
    if st.session_state.rag_engine:
        st.sidebar.subheader("📊 System Status")
        
        stats = st.session_state.rag_engine.get_stats()
        vector_stats = stats.get("vector_store_stats", {})
        
        st.sidebar.metric("Total Documents", vector_stats.get("total_documents", 0))
        st.sidebar.metric("Chat History", len(st.session_state.chat_history))
        
        if st.sidebar.button("🔄 Reset System"):
            st.session_state.rag_engine = None
            st.session_state.chat_history = []
            st.session_state.knowledge_base_loaded = False
            st.sidebar.success("System reset!")
    
    # Enhanced Help Section
    with st.sidebar.expander("🔍 Help & Troubleshooting"):
        st.markdown("""
        **Quick Start:**
        1. Enter your API key and endpoint
        2. Configure models and settings
        3. Click "Test Connection" first
        4. Click "Initialize System"
        5. Toggle RAG on/off as needed
        6. Configure MCP tools (when RAG is disabled)
        7. Start chatting!
        
        **Common Issues:**
        
        **404 Errors:**
        - Check endpoint URL format (must end with /v1)
        - Verify exact model names from deployment
        - Ensure models are deployed and running
        
        **Connection Issues:**
        - Verify API key permissions
        - Check network connectivity
        - Confirm firewall settings
        
        **Model Issues:**
        - Use exact model names (case-sensitive)
        - Check model deployment status
        - Verify embedding dimensions match
        
        **MCP Tools Features:**
        - **Web Search**: Real-time web information lookup
        - **Runtime Logs**: Application monitoring and debugging
        - **File Operations**: Read and write files safely
        - **Code Execution**: Execute code snippets (placeholder)
        - **Memory Management**: Save and retrieve information
        
        **Need Help?**
        - Contact your Nutanix administrator
        - Check Nutanix Enterprise AI documentation
        - Verify model availability in deployment
        """)

def rag_toggle_section():
    """Create the RAG toggle section."""
    # st.markdown('<div class="rag-toggle">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🔧 RAG Mode")
        if st.session_state.rag_enabled:
            st.write("**Enabled** - Chat with your documents using Retrieval-Augmented Generation")
        else:
            # Check if MCP tools are enabled
            mcp_tools_enabled = (hasattr(st.session_state.rag_engine, 'enable_mcp_tools') and 
                               st.session_state.rag_engine.enable_mcp_tools if st.session_state.rag_engine else False)
            
            if mcp_tools_enabled:
                st.write("**Disabled** - Direct chat with the language model enhanced by MCP tools")
                
                # Show enabled tools
                if st.session_state.rag_engine and st.session_state.rag_engine.mcp_tools_manager:
                    available_tools = st.session_state.rag_engine.mcp_tools_manager.get_available_tools()
                    st.write(f"🔧 **Active Tools:** {', '.join(available_tools)}")
            else:
                st.write("**Disabled** - Direct chat with the language model")
    
    with col2:
        rag_enabled = st.toggle("Enable RAG", value=st.session_state.rag_enabled)
        if rag_enabled != st.session_state.rag_enabled:
            st.session_state.rag_enabled = rag_enabled
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def file_upload_section():
    """Create the file upload section for RAG mode."""
    st.subheader("📁 Knowledge Base")
    
    # Sample Content
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Add Sample Content", use_container_width=True):
            with st.spinner("Adding sample Nutanix Enterprise AI documentation..."):
                try:
                    sample_text = """
                    Nutanix Enterprise AI provides a comprehensive artificial intelligence platform 
                    that enables organizations to deploy, manage, and scale AI workloads efficiently. 
                    
                    Key features include:
                    - Unified management of AI infrastructure
                    - Support for multiple AI frameworks including TensorFlow, PyTorch, and Hugging Face
                    - Scalable compute and storage resources with automatic scaling
                    - Enterprise-grade security and compliance features
                    - Integration with existing Nutanix infrastructure
                    - Built-in monitoring and observability tools
                    - Multi-cloud deployment capabilities
                    
                    The platform supports various AI use cases including:
                    - Natural language processing and understanding
                    - Computer vision and image recognition
                    - Predictive analytics and forecasting
                    - Machine learning model training and inference
                    - Conversational AI and chatbots
                    - Document processing and analysis
                    - Recommendation engines
                    
                    Nutanix Enterprise AI offers both pre-trained models and the ability to train custom models
                    using your organization's data. The platform provides APIs for easy integration with
                    existing applications and workflows.
                    """
                    
                    doc_ids = st.session_state.rag_engine.add_text(
                        sample_text,
                        source="nutanix_ai_overview",
                        metadata={
                            "type": "product_documentation",
                            "category": "nutanix_enterprise_ai",
                            "version": "1.0"
                        }
                    )
                    
                    st.success(f"✅ Added sample content ({len(doc_ids)} chunks)")
                    st.session_state.knowledge_base_loaded = True
                    
                except Exception as e:
                    st.error(f"❌ Error adding sample content: {str(e)}")
    
    with col2:
        if st.session_state.knowledge_base_loaded and st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            # Clear the vector store properly
            if st.session_state.rag_engine:
                st.session_state.rag_engine.clear_vector_store()
                st.session_state.knowledge_base_loaded = False
                st.success("✅ Knowledge base cleared!")
            else:
                st.error("❌ No RAG engine available to clear")
    
    # File Upload
    st.write("**Upload Your Documents:**")
    uploaded_files = st.file_uploader(
        "Choose files to add to your knowledge base",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx', 'md', 'py', 'js', 'html', 'json', 'csv'],
        help="Upload documents to expand your knowledge base"
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} file(s)")
        
        if st.button("📤 Process Files", type="primary", use_container_width=True):
            with st.spinner("Processing documents..."):
                processed_count = 0
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        # Save temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Process file
                        doc_ids = st.session_state.rag_engine.add_document_from_file(
                            tmp_file_path,
                            metadata={
                                "uploaded_by": "user",
                                "original_name": uploaded_file.name,
                                "upload_timestamp": datetime.now().isoformat()
                            }
                        )
                        processed_count += len(doc_ids)
                        
                        # Clean up
                        os.unlink(tmp_file_path)
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                progress_bar.progress(1.0)
                
                if processed_count > 0:
                    st.success(f"🎉 Processed {processed_count} document chunks!")
                    st.session_state.knowledge_base_loaded = True

def chat_interface():
    """Create the main chat interface."""
    # Chat header with mode information and controls
    col1, col2 = st.columns([4, 1])
    
    with col1:
        if st.session_state.rag_enabled and st.session_state.knowledge_base_loaded:
            st.markdown("#### 💬 Chat with Your Documents")
            st.info("🔍 **RAG Mode Active** - Your questions will be answered using uploaded documents")
        elif st.session_state.rag_enabled and not st.session_state.knowledge_base_loaded:
            st.markdown("#### 💬 Chat Interface")
            st.warning("📄 **RAG Mode** - Please upload documents first to enable document-based answers")
    
    with col2:
        if st.session_state.chat_history and st.button("🗑️ Clear Chat", help="Clear chat history", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Chat history container
    chat_history_container = st.container()
    
    with chat_history_container:
        # Display chat history using st.chat_message for better formatting
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        else:
            # Show helpful message when no chat history exists
            st.markdown("---")
            st.markdown("**💡 Start a conversation by typing your question below**")
            
            if st.session_state.rag_enabled and st.session_state.knowledge_base_loaded:
                st.markdown("**Try asking about your uploaded documents:**")
                st.markdown("- What are the main topics in the documents?")
                st.markdown("- Summarize the key points")
                st.markdown("- Ask specific questions about the content")
            else:
                # Check if MCP tools are enabled
                mcp_tools_enabled = (hasattr(st.session_state.rag_engine, 'enable_mcp_tools') and 
                                   st.session_state.rag_engine.enable_mcp_tools if st.session_state.rag_engine else False)
                
                if mcp_tools_enabled:
                    st.markdown("**Ask me anything - I have enhanced capabilities:**")
                    st.markdown("- 🌐 Web search for real-time information")
                    st.markdown("- 📊 Runtime logs and error checking")
                    st.markdown("- 📁 File operations (read/write)")
                    st.markdown("- 💻 Code execution assistance")
                    st.markdown("- 🧠 Memory management")
                    st.markdown("- General questions and discussions")
                else:
                    st.markdown("**Ask me anything:**")
                    st.markdown("- General questions")
                    st.markdown("- Technical discussions")
                    st.markdown("- Information requests")
            st.markdown("---")
    
    # Chat input at the bottom - outside the chat history container
    if prompt := st.chat_input("Ask a question...", key="chat_input"):
        # Add user message to history and display it
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response with streaming
        with st.chat_message("assistant"):
            try:
                if st.session_state.rag_enabled and st.session_state.knowledge_base_loaded:
                    # Use RAG with streaming
                    response_stream = st.session_state.rag_engine.ask_stream(prompt)
                else:
                    # Direct chat without RAG - check if MCP tools are enabled
                    if (hasattr(st.session_state.rag_engine, 'enable_mcp_tools') and 
                        st.session_state.rag_engine.enable_mcp_tools):
                        # Use MCP tools with streaming
                        response_stream = st.session_state.rag_engine.chat_with_tools_stream(prompt)
                    else:
                        # Direct chat without RAG or tools
                        response_stream = st.session_state.rag_engine.chat_stream(prompt, include_context=False)
                
                # Stream the response
                response = st.write_stream(response_stream)
                
                # Add assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                
                # Enhanced error handling for 404
                error_msg = str(e).lower()
                if '404' in error_msg or 'not found' in error_msg:
                    st.error("""
                    **🚨 404 Error - Model or Endpoint Not Found**
                    
                    This error usually means:
                    - **Wrong Chat Model Name**: The chat model doesn't exist in your Nutanix deployment
                    - **Incorrect Endpoint**: The API endpoint URL is wrong or inaccessible
                    - **Model Not Deployed**: The model isn't running or accessible
                    
                    **💡 Try these solutions:**
                    1. Go to the sidebar and click "🔍 Test Connection" to verify settings
                    2. Check the exact model names in your Nutanix deployment
                    3. Verify the endpoint URL format (should end with /v1)
                    4. Contact your Nutanix administrator for correct model names
                    """)
                elif 'connection' in error_msg or 'timeout' in error_msg:
                    st.error("""
                    **🌐 Connection Error**
                    
                    This error usually means:
                    - **Network Issues**: Can't reach the Nutanix cluster
                    - **Firewall Blocking**: Network security is blocking the connection
                    - **Service Down**: The Nutanix AI service might be temporarily unavailable
                    
                    **💡 Try these solutions:**
                    1. Check your network connection
                    2. Verify you can access the Nutanix cluster from your location
                    3. Contact your network administrator about firewall settings
                    """)
                elif 'auth' in error_msg or 'unauthorized' in error_msg or '401' in error_msg:
                    st.error("""
                    **🔒 Authentication Error**
                    
                    This error usually means:
                    - **Invalid API Key**: The API key is incorrect or expired
                    - **Insufficient Permissions**: The API key doesn't have access to the models
                    
                    **💡 Try these solutions:**
                    1. Verify your API key in the sidebar
                    2. Contact your Nutanix administrator to check API key permissions
                    3. Ensure the API key has access to the specified models
                    """)
                
                # Show debugging information in an expander
                with st.expander("🔧 Debug Information"):
                    stats = st.session_state.rag_engine.get_stats() if st.session_state.rag_engine else {}
                    st.write("**System Configuration:**")
                    st.write(f"- Endpoint: `{stats.get('base_url', 'Not set')}`")
                    st.write(f"- Chat Model: `{stats.get('model_name', 'Not set')}`")
                    st.write(f"- Endpoint Type: `{stats.get('endpoint_type', 'Unknown')}`")
                    st.write(f"- Temperature: `{stats.get('temperature', 'Not set')}`")
                    st.write(f"- RAG Mode: `{'Enabled' if st.session_state.rag_enabled else 'Disabled'}`")
                    st.write(f"- Knowledge Base: `{'Loaded' if st.session_state.knowledge_base_loaded else 'Not Loaded'}`")
                    st.write(f"- Error Type: `{type(e).__name__}`")

def main():
    """Main Streamlit application function."""
    initialize_session_state()
    
    # Sidebar configuration
    sidebar_configuration()
    
    # Main content area
    st.title("🌟 Nutanix Enterprise AI Chat")
    
    if not st.session_state.rag_engine:
        st.info("👈 Please configure and initialize the system using the sidebar")
        
        # Show configuration preview
        st.subheader("🔧 Quick Setup Guide")
        st.markdown("""
        1. **Enter API Key** - Your Nutanix Enterprise AI API key
        2. **Set Endpoint** - Your Nutanix cluster endpoint URL
        3. **Configure Models** - Embedding and chat model names
        4. **Initialize System** - Click the button to connect
        5. **Start Chatting** - Use RAG mode or direct chat
        """)
        
    else:
        # RAG Toggle Section
        rag_toggle_section()
        
        # Show appropriate interface based on RAG mode
        if st.session_state.rag_enabled:
            # For RAG mode, show file upload in an expander at the top
            with st.expander("📁 Knowledge Base Management", expanded=not st.session_state.knowledge_base_loaded):
                file_upload_section()
            
            # Chat interface takes full width
            if st.session_state.knowledge_base_loaded:
                chat_interface()
            else:
                st.info("📄 Please add documents to your knowledge base to start RAG-based chat")
                
                # Show sample questions
                st.subheader("❓ Sample Questions")
                st.markdown("""
                Once you add documents, you can ask questions like:
                - "What is Nutanix Enterprise AI?"
                - "What are the key features?"
                - "How does it support AI workloads?"
                """)
        else:
            # Direct chat mode
            chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by Nutanix Enterprise AI* 🌟")

if __name__ == "__main__":
    main() 