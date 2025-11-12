#!/usr/bin/env python3
"""
Simple launcher for University RAG Chatbot with local embeddings
"""

import os
import sys

def main():
    print("="*60)
    print("🎓 University of Sargodha RAG Chatbot")
    print("="*60)
    print()
    
    # Check if .env exists
    if not os.path.exists('.env'):
        print("📝 Creating .env file for local embeddings...")
        with open('.env', 'w') as f:
            f.write("# Local Embeddings Configuration (FREE!)\n")
            f.write("USE_LOCAL_EMBEDDINGS=true\n")
            f.write("EMBEDDING_MODEL=all-MiniLM-L6-v2\n")
            f.write("LLM_MODEL=gpt-3.5-turbo\n")
        print("✅ Created .env file\n")
    
    # Check vector store
    if not os.path.exists('models/faiss_db') or not os.listdir('models/faiss_db'):
        print("❌ Vector database not found!")
        print()
        print("Please run the setup first:")
        print("   python src/test_processing.py")
        print("   python src/test_chunking.py")
        print("   python src/test_vector_store.py")
        print()
        sys.exit(1)
    
    print("✅ Vector database found")
    print()
    print("🚀 Starting chatbot interface...")
    print("📍 Interface will open at: http://localhost:7860")
    print("⏸️  Press Ctrl+C to stop")
    print()
    print("-"*60)
    print()
    
    # Import and run
    try:
        # Add interface directory to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'interface'))
        
        # Run gradio app
        os.system('python interface/gradio_app.py')
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down... Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTry running manually:")
        print("   python interface/gradio_app.py")

if __name__ == "__main__":
    main()