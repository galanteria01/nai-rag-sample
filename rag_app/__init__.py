"""
RAG Application Package

A comprehensive Retrieval-Augmented Generation application that can interact with 
OpenAI-compatible APIs for creating embeddings and providing chat functionality.
"""

__version__ = "1.0.0"

from .embedding_service import EmbeddingService
from .vector_store import VectorStore, Document
from .document_processor import DocumentProcessor
from .rag_engine import RAGEngine

__all__ = ["EmbeddingService", "VectorStore", "Document", "DocumentProcessor", "RAGEngine"] 