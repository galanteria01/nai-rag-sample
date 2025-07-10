"""
RAG Engine for RAG Application

This module provides the main RAG (Retrieval-Augmented Generation) engine that
combines document retrieval with text generation using OpenAI-compatible APIs
including Nutanix Enterprise AI and other custom endpoints.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re

from openai import OpenAI
from .embedding_service import EmbeddingService
from .vector_store import VectorStore, Document
from .document_processor import DocumentProcessor


class RAGEngine:
    """
    Main RAG engine that combines retrieval and generation.
    Supports custom endpoints like Nutanix Enterprise AI.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        document_processor: DocumentProcessor,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_context_tokens: int = 8000,
        max_retrieved_docs: int = 5,
        custom_headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0
    ):
        """
        Initialize the RAG engine.
        
        Args:
            embedding_service: Service for creating embeddings
            vector_store: Vector store for document retrieval
            document_processor: Processor for handling documents
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API endpoint (defaults to OpenAI)
            model_name: Name of the chat model to use (can be any string for custom endpoints)
            temperature: Temperature for text generation
            max_context_tokens: Maximum tokens for context
            max_retrieved_docs: Maximum number of documents to retrieve
            custom_headers: Optional custom headers for API requests
            timeout: Request timeout in seconds
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.document_processor = document_processor
        self.model_name = model_name
        self.temperature = temperature
        self.max_context_tokens = max_context_tokens
        self.max_retrieved_docs = max_retrieved_docs
        self.custom_headers = custom_headers or {}
        self.timeout = timeout
        
        # Initialize OpenAI client for chat completion
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout
        }
        
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        # Add custom headers if provided
        if self.custom_headers:
            client_kwargs["default_headers"] = self.custom_headers
        
        self.client = OpenAI(**client_kwargs)
        
        # Chat history for conversation context
        self.chat_history: List[Dict[str, str]] = []
        
        # System prompt for the RAG assistant
        self.system_prompt = """You are a helpful AI assistant that answers questions based on the provided context documents. 

Instructions:
1. Use the context documents to answer the user's question as accurately as possible
2. If the answer isn't in the context documents, say so clearly
3. Cite the source documents when possible
4. Be concise but comprehensive
5. If multiple documents contain relevant information, synthesize the information appropriately

Context Documents:
{context}

Please answer the following question based on the context above."""
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to both embedding and chat APIs.
        
        Returns:
            Dictionary with connection test results
        """
        results = {
            "embedding_service": self.embedding_service.test_connection(),
            "chat_service": self._test_chat_connection()
        }
        
        results["overall_success"] = (
            results["embedding_service"]["success"] and 
            results["chat_service"]["success"]
        )
        
        return results
    
    def _test_chat_connection(self) -> Dict[str, Any]:
        """Test the chat API connection."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10,
                temperature=0
            )
            
            return {
                "success": True,
                "model": self.model_name,
                "base_url": self.base_url,
                "message": "Chat connection successful"
            }
        except Exception as e:
            return {
                "success": False,
                "model": self.model_name,
                "base_url": self.base_url,
                "error": str(e),
                "message": "Chat connection failed"
            }
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store with embeddings.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs that were added
        """
        added_ids = []
        
        for document in documents:
            if document.embedding is None:
                # Create embedding for the document
                document.embedding = self.embedding_service.create_embedding(document.content)
            
            self.vector_store.add_document(document)
            added_ids.append(document.id)
        
        return added_ids
    
    def add_document_from_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Process a file and add its documents to the vector store.
        
        Args:
            file_path: Path to the file to process
            metadata: Optional metadata to attach to documents
            
        Returns:
            List of document IDs that were added
        """
        # Process the file
        documents = self.document_processor.process_file(file_path, metadata)
        
        # Add documents to vector store
        return self.add_documents(documents)
    
    def add_documents_from_directory(
        self, 
        directory_path: str, 
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Process all files in a directory and add documents to the vector store.
        
        Args:
            directory_path: Path to directory to process
            recursive: Whether to process subdirectories
            file_patterns: Optional list of file patterns to match
            metadata: Optional metadata to attach to documents
            
        Returns:
            List of document IDs that were added
        """
        # Process the directory
        documents = self.document_processor.process_directory(
            directory_path, recursive, file_patterns, metadata
        )
        
        # Add documents to vector store
        return self.add_documents(documents)
    
    def add_text(self, text: str, source: str = "text_input", metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Process raw text and add it to the vector store.
        
        Args:
            text: Raw text content
            source: Source identifier for the text
            metadata: Optional metadata to attach to documents
            
        Returns:
            List of document IDs that were added
        """
        # Process the text
        documents = self.document_processor.process_text(text, source, metadata)
        
        # Add documents to vector store
        return self.add_documents(documents)
    
    def search_documents(
        self, 
        query: str, 
        k: int = None, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of documents to return (defaults to max_retrieved_docs)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if k is None:
            k = self.max_retrieved_docs
        
        return self.vector_store.search_by_text(
            query, self.embedding_service, k, filter_metadata
        )
    
    def chat(
        self, 
        message: str, 
        include_context: bool = True,
        k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        use_chat_history: bool = True,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Chat with the RAG assistant.
        
        Args:
            message: User message
            include_context: Whether to include retrieved context
            k: Number of documents to retrieve for context
            filter_metadata: Optional metadata filters for retrieval
            use_chat_history: Whether to use chat history for context
            max_tokens: Maximum tokens for response
            
        Returns:
            Dictionary with response and metadata
        """
        if k is None:
            k = self.max_retrieved_docs
        
        if max_tokens is None:
            max_tokens = 1000
        
        # Search for relevant documents
        retrieved_docs = []
        context_text = ""
        
        if include_context:
            retrieved_docs = self.search_documents(message, k, filter_metadata)
            
            # Build context from retrieved documents
            context_parts = []
            for i, (doc, score) in enumerate(retrieved_docs):
                source = doc.metadata.get("source", "Unknown")
                context_parts.append(f"Document {i+1} (Source: {source}, Score: {score:.3f}):\n{doc.content}")
            
            context_text = "\n\n".join(context_parts)
        
        # Build the prompt
        if include_context and context_text:
            system_message = self.system_prompt.format(context=context_text)
        else:
            system_message = "You are a helpful AI assistant. Answer the user's question to the best of your ability."
        
        # Prepare messages for chat completion
        messages = [{"role": "system", "content": system_message}]
        
        # Add chat history if enabled
        if use_chat_history:
            messages.extend(self.chat_history)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Truncate messages if they exceed context limit
        messages = self._truncate_messages(messages)
        
        try:
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            
            assistant_message = response.choices[0].message.content
            
            # Update chat history
            if use_chat_history:
                self.chat_history.append({"role": "user", "content": message})
                self.chat_history.append({"role": "assistant", "content": assistant_message})
                
                # Keep chat history manageable
                if len(self.chat_history) > 20:
                    self.chat_history = self.chat_history[-20:]
            
            return {
                "response": assistant_message,
                "retrieved_documents": [
                    {
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "similarity_score": score
                    }
                    for doc, score in retrieved_docs
                ],
                "context_used": bool(context_text),
                "model_used": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "endpoint_type": self._detect_endpoint_type()
            }
            
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "retrieved_documents": [],
                "context_used": False,
                "model_used": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "endpoint_type": self._detect_endpoint_type()
            }
    
    def chat_stream(
        self, 
        message: str, 
        include_context: bool = True,
        k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        use_chat_history: bool = True,
        max_tokens: Optional[int] = None
    ):
        """
        Chat with the RAG assistant using streaming responses.
        
        Args:
            message: User message
            include_context: Whether to include retrieved context
            k: Number of documents to retrieve for context
            filter_metadata: Optional metadata filters for retrieval
            use_chat_history: Whether to use chat history for context
            max_tokens: Maximum tokens for response
            
        Yields:
            Stream of response chunks
        """
        if k is None:
            k = self.max_retrieved_docs
        
        if max_tokens is None:
            max_tokens = 1000
        
        # Search for relevant documents
        retrieved_docs = []
        context_text = ""
        
        if include_context:
            retrieved_docs = self.search_documents(message, k, filter_metadata)
            
            # Build context from retrieved documents
            context_parts = []
            for i, (doc, score) in enumerate(retrieved_docs):
                source = doc.metadata.get("source", "Unknown")
                context_parts.append(f"Document {i+1} (Source: {source}, Score: {score:.3f}):\n{doc.content}")
            
            context_text = "\n\n".join(context_parts)
        
        # Build the prompt
        if include_context and context_text:
            system_message = self.system_prompt.format(context=context_text)
        else:
            system_message = "You are a helpful AI assistant. Answer the user's question to the best of your ability."
        
        # Prepare messages for chat completion
        messages = [{"role": "system", "content": system_message}]
        
        # Add chat history if enabled
        if use_chat_history:
            messages.extend(self.chat_history)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Truncate messages if they exceed context limit
        messages = self._truncate_messages(messages)
        
        # Generate streaming response
        full_response = ""
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # Update chat history
            if use_chat_history:
                self.chat_history.append({"role": "user", "content": message})
                self.chat_history.append({"role": "assistant", "content": full_response})
                
                # Keep chat history manageable
                if len(self.chat_history) > 20:
                    self.chat_history = self.chat_history[-20:]
                    
        except Exception as e:
            yield f"Error generating response: {str(e)}"

    def ask(self, question: str, **kwargs) -> str:
        """
        Simple question-answering interface.
        
        Args:
            question: Question to ask
            **kwargs: Additional arguments for chat method
            
        Returns:
            Answer string
        """
        result = self.chat(question, **kwargs)
        return result["response"]
    
    def ask_stream(self, question: str, **kwargs):
        """
        Simple question-answering interface with streaming.
        
        Args:
            question: Question to ask
            **kwargs: Additional arguments for chat_stream method
            
        Yields:
            Stream of response chunks
        """
        yield from self.chat_stream(question, **kwargs)
    
    def clear_chat_history(self):
        """Clear the chat history."""
        self.chat_history = []
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the current chat history."""
        return self.chat_history.copy()
    
    def _truncate_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Truncate messages to fit within context limit."""
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        total_chars = sum(len(msg["content"]) for msg in messages)
        estimated_tokens = total_chars // 4
        
        if estimated_tokens <= self.max_context_tokens:
            return messages
        
        # Keep system message and recent messages
        if len(messages) <= 2:
            return messages
        
        # Keep system message (first) and gradually remove older messages
        system_msg = messages[0]
        other_messages = messages[1:]
        
        while len(other_messages) > 1:
            # Remove second message (oldest non-system message)
            other_messages = other_messages[1:]
            
            # Recalculate tokens
            truncated_messages = [system_msg] + other_messages
            total_chars = sum(len(msg["content"]) for msg in truncated_messages)
            estimated_tokens = total_chars // 4
            
            if estimated_tokens <= self.max_context_tokens:
                break
        
        return [system_msg] + other_messages
    
    def _detect_endpoint_type(self) -> str:
        """Detect the type of endpoint based on base URL."""
        if not self.base_url:
            return "OpenAI"
        
        base_url_lower = self.base_url.lower()
        
        if "nutanix" in base_url_lower:
            return "Nutanix Enterprise AI"
        elif "azure" in base_url_lower:
            return "Azure OpenAI"
        elif "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
            return "Local API"
        elif "api.openai.com" in base_url_lower:
            return "OpenAI"
        else:
            return "Custom Endpoint"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG engine.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_context_tokens": self.max_context_tokens,
            "max_retrieved_docs": self.max_retrieved_docs,
            "chat_history_length": len(self.chat_history),
            "base_url": self.base_url,
            "endpoint_type": self._detect_endpoint_type(),
            "custom_headers": bool(self.custom_headers),
            "timeout": self.timeout,
            "vector_store_stats": self.vector_store.get_stats(),
            "embedding_service_info": self.embedding_service.get_model_info(),
            "document_processor_stats": self.document_processor.get_processing_stats()
        }
    
    def save_knowledge_base(self, filepath: str):
        """
        Save the knowledge base (vector store) to disk.
        
        Args:
            filepath: Path to save the knowledge base
        """
        self.vector_store.save(filepath)
    
    def load_knowledge_base(self, filepath: str):
        """
        Load a knowledge base (vector store) from disk.
        
        Args:
            filepath: Path to the saved knowledge base
        """
        self.vector_store = VectorStore.load(filepath)
    
    def export_chat_history(self, filepath: str):
        """
        Export chat history to a JSON file.
        
        Args:
            filepath: Path to save the chat history
        """
        with open(filepath, 'w') as f:
            json.dump(self.chat_history, f, indent=2)
    
    def import_chat_history(self, filepath: str):
        """
        Import chat history from a JSON file.
        
        Args:
            filepath: Path to the chat history file
        """
        with open(filepath, 'r') as f:
            self.chat_history = json.load(f)
    
    def create_context_summary(self, documents: List[Tuple[Document, float]]) -> str:
        """
        Create a summary of the retrieved context documents.
        
        Args:
            documents: List of (document, similarity_score) tuples
            
        Returns:
            Context summary string
        """
        if not documents:
            return "No relevant documents found."
        
        summary_parts = []
        for i, (doc, score) in enumerate(documents):
            source = doc.metadata.get("source", "Unknown")
            file_type = doc.metadata.get("file_type", "Unknown")
            
            summary_parts.append(
                f"Document {i+1}: {source} ({file_type}) - Similarity: {score:.3f}"
            )
        
        return "\n".join(summary_parts)
    
    def update_system_prompt(self, new_prompt: str):
        """
        Update the system prompt for the RAG assistant.
        
        Args:
            new_prompt: New system prompt (should include {context} placeholder)
        """
        self.system_prompt = new_prompt
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        return self.vector_store.get_document(doc_id)
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """
        List documents in the knowledge base.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of documents
        """
        return self.vector_store.list_documents(limit, offset)
    
    def remove_document(self, doc_id: str):
        """
        Remove a document from the knowledge base.
        
        Args:
            doc_id: Document ID to remove
        """
        self.vector_store.remove_document(doc_id)
    
    @classmethod
    def create_for_nutanix(
        cls,
        api_key: str,
        base_url: str,
        embedding_model: str,
        chat_model: str,
        embedding_dimension: int,
        **kwargs
    ) -> "RAGEngine":
        """
        Convenience method to create a RAG engine for Nutanix Enterprise AI.
        
        Args:
            api_key: Nutanix API key
            base_url: Nutanix endpoint URL
            embedding_model: Embedding model name in Nutanix
            chat_model: Chat model name in Nutanix
            embedding_dimension: Dimension of embeddings
            **kwargs: Additional arguments
            
        Returns:
            Configured RAGEngine instance
        """
        # Extract parameters for embedding service
        embedding_service_kwargs = {
            "max_tokens": kwargs.get("max_tokens", 8192),
            "custom_headers": kwargs.get("custom_headers", None),
            "timeout": kwargs.get("timeout", 60.0)
        }
        
        # Create embedding service
        embedding_service = EmbeddingService.create_for_nutanix(
            api_key=api_key,
            base_url=base_url,
            model_name=embedding_model,
            **embedding_service_kwargs
        )
        
        # Create vector store
        vector_store = VectorStore(
            dimension=embedding_dimension,
            index_type=kwargs.get("index_type", "flat")
        )
        
        # Create document processor
        document_processor = DocumentProcessor(
            chunk_size=kwargs.get("chunk_size", 1000),
            chunk_overlap=kwargs.get("chunk_overlap", 200),
            chunking_strategy=kwargs.get("chunking_strategy", "recursive"),
            max_tokens_per_chunk=kwargs.get("max_tokens_per_chunk", None)
        )
        
        # Extract only the parameters that RAGEngine constructor accepts
        rag_engine_kwargs = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_context_tokens": kwargs.get("max_context_tokens", 8000),
            "max_retrieved_docs": kwargs.get("max_retrieved_docs", 5),
            "custom_headers": kwargs.get("custom_headers", None),
            "timeout": kwargs.get("timeout", 60.0)
        }
        
        # Create RAG engine
        return cls(
            embedding_service=embedding_service,
            vector_store=vector_store,
            document_processor=document_processor,
            api_key=api_key,
            base_url=base_url,
            model_name=chat_model,
            **rag_engine_kwargs
        ) 