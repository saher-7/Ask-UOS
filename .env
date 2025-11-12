#!/usr/bin/env python3
"""
Test Local Embeddings Setup - Verify everything works without OpenAI
"""

import os
import sys

def test_imports():
    """Test if all required packages are installed"""
    print("📦 Testing imports...")
    
    try:
        import sentence_transformers
        print("   ✅ sentence-transformers")
    except ImportError:
        print("   ❌ sentence-transformers - Run: pip install sentence-transformers")
        return False
    
    try:
        import faiss
        print("   ✅ faiss-cpu")
    except ImportError:
        print("   ❌ faiss-cpu - Run: pip install faiss-cpu")
        return False
    
    try:
        import numpy
        print("   ✅ numpy")
    except ImportError:
        print("   ❌ numpy - Run: pip install numpy")
        return False
    
    try:
        import gradio
        print("   ✅ gradio")
    except ImportError:
        print("   ❌ gradio - Run: pip install gradio")
        return False
    
    print()
    return True

def test_local_embeddings():
    """Test local embedding service"""
    print("🧪 Testing local embedding service...")
    
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.embedding_service import LocalEmbeddingService
        
        print("   Loading model (this may take a minute first time)...")
        service = LocalEmbeddingService(model_name="all-MiniLM-L6-v2")
        
        print("   Creating test embedding...")
        test_text = "University of Sargodha admission rules"
        embedding = service.create_embedding(test_text)
        
        if embedding and len(embedding) > 0:
            print(f"   ✅ Embedding created successfully! (dimension: {len(embedding)})")
            return True
        else:
            print("   ❌ Failed to create embedding")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_vector_store():
    """Test if vector store exists"""
    print("\n🗄️  Checking vector store...")
    
    vector_db_path = 'models/faiss_db'
    
    if not os.path.exists(vector_db_path):
        print("   ❌ Vector database not found")
        print("   📝 Run: python src/test_vector_store.py")
        return False
    
    files = os.listdir(vector_db_path)
    if not files:
        print("   ❌ Vector database is empty")
        print("   📝 Run: python src/test_vector_store.py")
        return False
    
    print(f"   ✅ Vector database found with {len(files)} files")
    return True

def test_rag_pipeline():
    """Test RAG pipeline with local embeddings"""
    print("\n🔧 Testing RAG pipeline...")
    
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.rag_pipeline import UniversityRAGPipeline
        from src.vector_store import UniversityVectorStore
        
        print("   Initializing vector store...")
        vector_store = UniversityVectorStore()
        stats = vector_store.get_collection_stats()
        
        if stats.get('total_documents', 0) == 0:
            print("   ❌ No documents in vector store")
            print("   📝 Run: python src/test_vector_store.py")
            return False
        
        print(f"   ✅ Vector store has {stats['total_documents']} documents")
        
        print("   Initializing RAG pipeline with local embeddings...")
        rag = UniversityRAGPipeline(
            vector_store=vector_store,
            use_local_embeddings=True
        )
        
        print("   Testing query...")
        response = rag.query("What are the admission requirements?")
        
        if response and response.answer:
            print(f"   ✅ Query successful!")
            print(f"   📝 Response preview: {response.answer[:100]}...")
            return True
        else:
            print("   ❌ Query failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("🚀 Local Embeddings Setup Test")
    print("="*60)
    print()
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Install missing packages first.")
        return
    
    # Test 2: Local Embeddings
    if not test_local_embeddings():
        print("\n❌ Local embeddings test failed.")
        return
    
    # Test 3: Vector Store
    if not test_vector_store():
        print("\n❌ Vector store test failed.")
        return
    
    # Test 4: RAG Pipeline
    if not test_rag_pipeline():
        print("\n❌ RAG pipeline test failed.")
        return
    
    # Success!
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print()
    print("🎉 Your system is ready to run with local embeddings!")
    print()
    print("🚀 Start the chatbot:")
    print("   python interface/gradio_app.py")
    print()
    print("⚠️  Note: Without OpenAI API, responses will be simpler")
    print("   but the interface will work perfectly for testing!")
    print()

if __name__ == "__main__":
    main()