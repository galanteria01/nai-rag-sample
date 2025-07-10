"""
Vector Store for RAG Application

This module provides a vector store using FAISS for efficient similarity search
and storage of embeddings with metadata.
"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Document:
    """
    Represents a document with its content and metadata.
    """
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class VectorStore:
    """
    A vector store for storing and searching document embeddings using FAISS.
    """
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.documents: Dict[str, Document] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.current_index = 0
        
        # Initialize FAISS index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the FAISS index based on the specified type."""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, min(100, max(1, len(self.documents) // 10)))
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def add_document(self, document: Document):
        """
        Add a document to the vector store.
        
        Args:
            document: Document to add
        """
        if document.embedding is None:
            raise ValueError("Document must have an embedding")
        
        # Normalize embedding for cosine similarity
        normalized_embedding = self._normalize_embedding(document.embedding)
        
        # Add to FAISS index
        self.index.add(normalized_embedding.reshape(1, -1))
        
        # Update mappings
        self.documents[document.id] = document
        self.id_to_index[document.id] = self.current_index
        self.index_to_id[self.current_index] = document.id
        self.current_index += 1
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained and len(self.documents) >= 100:
            embeddings = np.array([doc.embedding for doc in self.documents.values()])
            normalized_embeddings = self._normalize_embeddings(embeddings)
            self.index.train(normalized_embeddings)
    
    def add_documents(self, documents: List[Document]):
        """
        Add multiple documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        for document in documents:
            self.add_document(document)
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if len(self.documents) == 0:
            return []
        
        # Normalize query embedding
        normalized_query = self._normalize_embedding(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(normalized_query.reshape(1, -1), min(k * 2, len(self.documents)))
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # Invalid index
                continue
            
            doc_id = self.index_to_id[idx]
            document = self.documents[doc_id]
            
            # Apply metadata filters if provided
            if filter_metadata:
                if not self._matches_filter(document.metadata, filter_metadata):
                    continue
            
            results.append((document, float(score)))
            
            if len(results) >= k:
                break
        
        return results
    
    def search_by_text(
        self, 
        query_text: str, 
        embedding_service,
        k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents using text query.
        
        Args:
            query_text: Query text
            embedding_service: Embedding service to create query embedding
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (document, similarity_score) tuples
        """
        query_embedding = embedding_service.create_embedding(query_text)
        return self.search(query_embedding, k, filter_metadata)
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        return self.documents.get(doc_id)
    
    def update_document(self, document: Document):
        """
        Update an existing document.
        
        Args:
            document: Updated document
        """
        if document.id not in self.documents:
            raise ValueError(f"Document {document.id} not found")
        
        # For simplicity, we'll remove and re-add the document
        self.remove_document(document.id)
        self.add_document(document)
    
    def remove_document(self, doc_id: str):
        """
        Remove a document from the vector store.
        
        Args:
            doc_id: Document ID to remove
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document {doc_id} not found")
        
        # Remove from mappings
        del self.documents[doc_id]
        index_pos = self.id_to_index[doc_id]
        del self.id_to_index[doc_id]
        del self.index_to_id[index_pos]
        
        # For FAISS, we need to rebuild the index to actually remove the vector
        # This is a limitation of FAISS - it doesn't support efficient removal
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild the FAISS index (used when removing documents)."""
        if not self.documents:
            self._initialize_index()
            return
        
        # Reinitialize index
        self._initialize_index()
        
        # Re-add all documents
        embeddings = []
        doc_ids = []
        
        for doc_id, document in self.documents.items():
            embeddings.append(document.embedding)
            doc_ids.append(doc_id)
        
        if embeddings:
            normalized_embeddings = self._normalize_embeddings(np.array(embeddings))
            self.index.add(normalized_embeddings)
            
            # Update mappings
            self.id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}
            self.index_to_id = {i: doc_id for i, doc_id in enumerate(doc_ids)}
            self.current_index = len(doc_ids)
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize a single embedding for cosine similarity."""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize multiple embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Check if document metadata matches the filter criteria."""
        for key, value in filter_metadata.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": getattr(self.index, 'is_trained', True),
            "memory_usage_mb": self.index.compute_memory_usage() / 1024 / 1024 if hasattr(self.index, 'compute_memory_usage') else None
        }
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """
        List documents in the vector store.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of documents
        """
        all_docs = list(self.documents.values())
        return all_docs[offset:offset + limit]
    
    def save(self, filepath: str):
        """
        Save the vector store to disk.
        
        Args:
            filepath: Path to save the vector store
        """
        # Create directory if it doesn't exist (only if filepath has a directory)
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "documents": {},
            "id_to_index": self.id_to_index,
            "index_to_id": self.index_to_id,
            "current_index": self.current_index
        }
        
        # Convert documents to serializable format
        for doc_id, document in self.documents.items():
            metadata["documents"][doc_id] = {
                "id": document.id,
                "content": document.content,
                "metadata": document.metadata,
                "embedding": document.embedding.tolist() if document.embedding is not None else None,
                "created_at": document.created_at.isoformat() if document.created_at else None
            }
        
        with open(f"{filepath}.metadata", "w") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "VectorStore":
        """
        Load a vector store from disk.
        
        Args:
            filepath: Path to the saved vector store
            
        Returns:
            Loaded vector store
        """
        # Load metadata
        with open(f"{filepath}.metadata", "r") as f:
            metadata = json.load(f)
        
        # Create vector store
        vector_store = cls(metadata["dimension"], metadata["index_type"])
        
        # Load FAISS index
        vector_store.index = faiss.read_index(f"{filepath}.index")
        
        # Restore documents
        for doc_id, doc_data in metadata["documents"].items():
            document = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                embedding=np.array(doc_data["embedding"]) if doc_data["embedding"] else None,
                created_at=datetime.fromisoformat(doc_data["created_at"]) if doc_data["created_at"] else None
            )
            vector_store.documents[doc_id] = document
        
        # Restore mappings
        vector_store.id_to_index = {k: int(v) for k, v in metadata["id_to_index"].items()}
        vector_store.index_to_id = {int(k): v for k, v in metadata["index_to_id"].items()}
        vector_store.current_index = metadata["current_index"]
        
        return vector_store 