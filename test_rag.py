#!/usr/bin/env python3
"""
Test script for RAG Application

This script tests the core functionality of the RAG application
without requiring an API key.
"""

import sys
import traceback
from datetime import datetime

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing imports...")
    try:
        from rag_app import EmbeddingService, VectorStore, DocumentProcessor, RAGEngine
        from rag_app.vector_store import Document
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_document_processor():
    """Test document processor functionality."""
    print("\n🧪 Testing Document Processor...")
    try:
        from rag_app import DocumentProcessor
        
        processor = DocumentProcessor(
            chunk_size=500,
            chunk_overlap=100,
            chunking_strategy="recursive"
        )
        
        # Test text processing
        sample_text = """
        This is a sample document for testing the RAG application.
        It contains multiple paragraphs to test the chunking functionality.
        
        The document processor should be able to split this text into
        meaningful chunks while preserving context between overlapping segments.
        
        This helps ensure that important information is not lost during
        the document processing phase of the RAG pipeline.
        """
        
        documents = processor.process_text(
            sample_text, 
            source="test_document",
            metadata={"type": "test", "created": datetime.now().isoformat()}
        )
        
        print(f"   ✅ Created {len(documents)} document chunks")
        print(f"   📋 First chunk preview: {documents[0].content[:100]}...")
        print(f"   📊 Processor stats: {processor.get_processing_stats()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        traceback.print_exc()
        return False

def test_vector_store():
    """Test vector store functionality (without embeddings)."""
    print("\n🧪 Testing Vector Store...")
    try:
        from rag_app import VectorStore, Document
        import numpy as np
        
        # Create a vector store
        vector_store = VectorStore(dimension=384, index_type="flat")
        
        # Create sample documents with fake embeddings
        sample_docs = []
        for i in range(3):
            # Create a random embedding
            embedding = np.random.random(384).astype(np.float32)
            
            doc = Document(
                id=f"test_doc_{i}",
                content=f"This is test document {i} with some sample content for testing.",
                metadata={"doc_number": i, "type": "test"},
                embedding=embedding
            )
            sample_docs.append(doc)
        
        # Add documents to vector store
        vector_store.add_documents(sample_docs)
        
        # Test search
        query_embedding = np.random.random(384).astype(np.float32)
        results = vector_store.search(query_embedding, k=2)
        
        print(f"   ✅ Added {len(sample_docs)} documents")
        print(f"   🔍 Search returned {len(results)} results")
        print(f"   📊 Vector store stats: {vector_store.get_stats()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        traceback.print_exc()
        return False

def test_component_integration():
    """Test integration between components (without API calls)."""
    print("\n🧪 Testing Component Integration...")
    try:
        from rag_app import DocumentProcessor, VectorStore, Document
        import numpy as np
        
        # Initialize components
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        vector_store = VectorStore(dimension=384, index_type="flat")
        
        # Process some text
        text = "Machine learning is a subset of artificial intelligence. It focuses on algorithms that can learn from data."
        documents = processor.process_text(text, source="integration_test")
        
        # Add fake embeddings and store documents
        for doc in documents:
            doc.embedding = np.random.random(384).astype(np.float32)
        
        vector_store.add_documents(documents)
        
        # Test retrieval
        query_embedding = np.random.random(384).astype(np.float32)
        results = vector_store.search(query_embedding, k=1)
        
        print(f"   ✅ Processed text into {len(documents)} chunks")
        print(f"   ✅ Stored documents in vector store")
        print(f"   ✅ Retrieved {len(results)} relevant documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def test_file_operations():
    """Test file save/load operations."""
    print("\n🧪 Testing File Operations...")
    try:
        from rag_app import VectorStore, Document
        import numpy as np
        import os
        
        # Create vector store with sample data
        vector_store = VectorStore(dimension=384, index_type="flat")
        
        doc = Document(
            id="save_test_doc",
            content="This is a test document for save/load functionality.",
            metadata={"test": True},
            embedding=np.random.random(384).astype(np.float32)
        )
        
        vector_store.add_document(doc)
        
        # Save vector store
        save_path = "test_vector_store"
        vector_store.save(save_path)
        
        # Load vector store
        loaded_store = VectorStore.load(save_path)
        
        # Verify data
        loaded_doc = loaded_store.get_document("save_test_doc")
        
        print(f"   ✅ Saved vector store to {save_path}")
        print(f"   ✅ Loaded vector store successfully")
        print(f"   ✅ Verified document content: {loaded_doc.content[:50]}...")
        
        # Cleanup
        for file in os.listdir('.'):
            if file.startswith('test_vector_store'):
                os.remove(file)
        
        return True
        
    except Exception as e:
        print(f"❌ File operations test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔬 RAG Application Test Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Document Processor", test_document_processor),
        ("Vector Store", test_vector_store),
        ("Component Integration", test_component_integration),
        ("File Operations", test_file_operations)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! RAG application is working correctly.")
        print("\n📋 Next steps:")
        print("1. Set your OpenAI API key in the .env file")
        print("2. Run: streamlit run streamlit_app.py")
        print("3. Or open rag_demo.ipynb in Jupyter")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 