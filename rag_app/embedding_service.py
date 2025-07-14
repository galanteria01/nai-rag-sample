"""
Embedding Service for RAG Application

This module provides an embedding service that can work with OpenAI-compatible APIs
including Nutanix Enterprise AI and other custom endpoints.
"""

import os
from typing import List, Optional, Dict, Any
import numpy as np
from openai import OpenAI
import tiktoken


class EmbeddingService:
    """
    A service for creating embeddings using OpenAI-compatible APIs.
    Supports custom endpoints like Nutanix Enterprise AI.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "text-embedding-3-small",
        max_tokens: int = 8192,
        custom_headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0
    ):
        """
        Initialize the embedding service.
        
        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API endpoint (defaults to OpenAI)
            model_name: Name of the embedding model to use (can be any string for custom endpoints)
            max_tokens: Maximum number of tokens per request
            custom_headers: Optional custom headers for API requests
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("NUTANIX_API_KEY")
        self.base_url = base_url or os.getenv("NUTANIX_ENDPOINT")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.custom_headers = custom_headers or {}
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
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
        
        # Initialize tokenizer - try model-specific first, then fallback
        self.tokenizer = self._get_tokenizer()
        
        # Cache for embedding dimension
        self._embedding_dimension = None
    
    def _get_tokenizer(self):
        """Get appropriate tokenizer for the model."""
        try:
            # Try to get tokenizer for the specific model
            return tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # For custom models or unknown models, use a default tokenizer
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                # Last resort fallback
                return tiktoken.get_encoding("gpt2")
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback: rough estimation
            return len(text) // 4
    
    def truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Input text
            max_tokens: Maximum tokens (defaults to self.max_tokens)
            
        Returns:
            Truncated text
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        try:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens)
        except Exception:
            # Fallback: character-based truncation
            estimated_chars = max_tokens * 4
            return text[:estimated_chars] if len(text) > estimated_chars else text
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create an embedding for a single text string.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Truncate text if it's too long
        text = self.truncate_text(text)
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                encoding_format="float"
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
            
        except Exception as e:
            raise RuntimeError(f"Failed to create embedding: {str(e)}")
    
    def create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for multiple text strings.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        for text in texts:
            embedding = self.create_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        Create embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Truncate texts in batch
            truncated_batch = [self.truncate_text(text) for text in batch]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=truncated_batch,
                    encoding_format="float"
                )
                
                batch_embeddings = [
                    np.array(item.embedding, dtype=np.float32) 
                    for item in response.data
                ]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                raise RuntimeError(f"Failed to create batch embeddings: {str(e)}")
        
        return all_embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            Embedding dimension
        """
        if self._embedding_dimension is None:
            # Create a test embedding to get dimension
            test_embedding = self.create_embedding("test")
            self._embedding_dimension = len(test_embedding)
        
        return self._embedding_dimension
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            a: First embedding vector
            b: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the API endpoint.
        
        Returns:
            Dictionary with connection test results
        """
        try:
            # Try to create a simple embedding
            test_embedding = self.create_embedding("connection test")
            return {
                "success": True,
                "model": self.model_name,
                "base_url": self.base_url,
                "embedding_dimension": len(test_embedding),
                "message": "Connection successful"
            }
        except Exception as e:
            return {
                "success": False,
                "model": self.model_name,
                "base_url": self.base_url,
                "error": str(e),
                "message": "Connection failed"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model and configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "embedding_dimension": self.get_embedding_dimension() if self._embedding_dimension else "Unknown",
            "api_key_set": bool(self.api_key),
            "custom_headers": bool(self.custom_headers),
            "timeout": self.timeout,
            "endpoint_type": self._detect_endpoint_type()
        }
    
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
    
    @classmethod
    def create_for_nutanix(
        cls,
        api_key: str,
        base_url: str,
        model_name: str,
        **kwargs
    ) -> "EmbeddingService":
        """
        Convenience method to create an embedding service for Nutanix Enterprise AI.
        
        Args:
            api_key: Nutanix API key
            base_url: Nutanix endpoint URL
            model_name: Model name in Nutanix
            **kwargs: Additional arguments
            
        Returns:
            Configured EmbeddingService instance
        """
        return cls(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            **kwargs
        )
    
    @classmethod
    def create_for_azure(
        cls,
        api_key: str,
        base_url: str,
        model_name: str,
        api_version: str = "2023-05-15",
        **kwargs
    ) -> "EmbeddingService":
        """
        Convenience method to create an embedding service for Azure OpenAI.
        
        Args:
            api_key: Azure API key
            base_url: Azure endpoint URL
            model_name: Model deployment name
            api_version: API version
            **kwargs: Additional arguments
            
        Returns:
            Configured EmbeddingService instance
        """
        # Add API version to headers for Azure
        custom_headers = kwargs.get("custom_headers", {})
        custom_headers["api-version"] = api_version
        kwargs["custom_headers"] = custom_headers
        
        return cls(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            **kwargs
        ) 