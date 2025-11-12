#!/usr/bin/env python3
"""
Quick Fix Script for University RAG Chatbot
Run this to apply all fixes and test the system
"""

import os
import sys
from typing import Dict, Any  # ✅ Added (required for type hints in get_pipeline_stats)


def create_directories():
    """Create necessary directories"""
    dirs = [
        'logs', 'models', 'models/faiss_db', 'models/embedding_cache',
        'data', 'data/pdfs', 'data/web_content', 'data/processed'
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created/verified: {dir_path}")


def add_methods_to_rag_pipeline():
    """Add missing methods to rag_pipeline.py"""
    rag_file = 'src/rag_pipeline.py'

    if not os.path.exists(rag_file):
        print(f"❌ File not found: {rag_file}")
        return False

    with open(rag_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if methods already exist
    if 'def query(' in content and 'def get_pipeline_stats(' in content:
        print(f"✅ {rag_file} already has required methods")
        return True

    # ✅ Use consistent indentation (4 spaces) and no leading blank lines
    methods_to_add = '''
    def query(self, query: str, include_sources: bool = True) -> RAGResponse:
        """Alias for process_query to maintain compatibility with interfaces"""
        return self.process_query(query, include_sources)

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics - alias for get_statistics"""
        stats = self.get_statistics()

        # Flatten structure for easier access
        return {
            'queries': {
                'total_queries': stats['pipeline_stats']['total_queries'],
                'successful_responses': stats['pipeline_stats']['successful_responses'],
                'success_rate': float(str(stats['pipeline_stats']['success_rate']).rstrip('%'))
            },
            'performance': {
                'average_response_time': float(str(stats['pipeline_stats']['average_response_time']).rstrip('s')),
                'total_tokens_used': stats['pipeline_stats']['total_tokens_used'],
                'average_tokens_per_query': stats['pipeline_stats']['total_tokens_used'] // max(1, stats['pipeline_stats']['total_queries'])
            },
            'configuration': stats['configuration'],
            'vector_store': stats.get('vector_store_stats', {})
        }
    '''

    lines = content.split('\n')
    insert_index = -1

    for i in range(len(lines) - 1, 0, -1):
        if lines[i].strip().startswith('def ') and not lines[i].strip().startswith('def __'):
            for j in range(i + 1, len(lines)):
                if lines[j].strip() and not lines[j].startswith((' ', '\t')):
                    insert_index = j
                    break
            if insert_index > 0:
                break

    if insert_index > 0:
        lines.insert(insert_index, methods_to_add)
        content = '\n'.join(lines)

        with open(rag_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ Added missing methods to {rag_file}")
        return True
    else:
        print(f"❌ Could not find insertion point in {rag_file}")
        return False


def check_env_file():
    """Check if .env file exists"""
    if not os.path.exists('.env'):
        print("⚠️  .env file not found. Creating template...")

        env_template = """# OpenAI API Key (required for embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Groq API Key for LLM (alternative to OpenAI for chat)
GROQ_API_KEY=your_groq_api_key_here

# Model configurations
EMBEDDING_MODEL=text-embedding-ada-002
LLM_MODEL=gpt-3.5-turbo
"""

        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_template)

        print("✅ Created .env template. Please add your API keys!")
        print("📝 Edit .env file and add your OPENAI_API_KEY")
        return False
    else:
        print("✅ .env file found")

        with open('.env', 'r', encoding='utf-8') as f:
            env_content = f.read()

        if 'OPENAI_API_KEY' not in env_content or 'your_openai_api_key_here' in env_content:
            print("⚠️  Please add your OPENAI_API_KEY to .env file")
            return False

        return True


def check_data():
    """Check if data exists"""
    pdf_folder = 'data/pdfs'
    web_folder = 'data/web_content'
    processed_folder = 'data/processed'

    has_pdfs = os.path.exists(pdf_folder) and len(os.listdir(pdf_folder)) > 0
    has_web = os.path.exists(web_folder) and len(os.listdir(web_folder)) > 0
    has_processed = os.path.exists(processed_folder) and len(os.listdir(processed_folder)) > 0

    if not (has_pdfs or has_web):
        print("⚠️  No source documents found!")
        print("📝 Please run: python src/collect_documents.py")
        return False

    if not has_processed:
        print("⚠️  No processed documents found!")
        print("📝 Please run: python src/test_processing.py")
        return False

    print("✅ Source documents found")
    return True


def check_vector_store():
    """Check if vector store exists"""
    vector_db_path = 'models/faiss_db'

    if not os.path.exists(vector_db_path) or not os.listdir(vector_db_path):
        print("⚠️  Vector database not found!")
        print("📝 You need to run the setup pipeline:")
        print("   1. python src/test_processing.py")
        print("   2. python src/test_chunking.py")
        print("   3. python src/test_vector_store.py")
        return False

    print("✅ Vector database found")
    return True


def main():
    print("=" * 60)
    print("🔧 University RAG Chatbot - Quick Fix Script")
    print("=" * 60)
    print()

    print("📁 Step 1: Creating directories...")
    create_directories()
    print()

    print("🔧 Step 2: Fixing code...")
    add_methods_to_rag_pipeline()
    print()

    print("🔑 Step 3: Checking environment variables...")
    env_ok = check_env_file()
    print()

    print("📚 Step 4: Checking data...")
    data_ok = check_data()
    print()

    print("🗄️  Step 5: Checking vector store...")
    vector_ok = check_vector_store()
    print()

    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    if env_ok and data_ok and vector_ok:
        print("✅ System is ready to run!")
        print()
        print("🚀 To start the chatbot:")
        print("   python interface/gradio_app.py")
        print()
        print("   OR")
        print()
        print("   python interface/streamlit_app.py")
    else:
        print("⚠️  System needs setup. Please:")
        print()
        if not env_ok:
            print("   1. Add your OPENAI_API_KEY to .env file")
        if not data_ok:
            print("   2. Run: python src/collect_documents.py")
            print("      Run: python src/test_processing.py")
        if not vector_ok:
            print("   3. Run: python src/test_chunking.py")
            print("      Run: python src/test_vector_store.py")
        print()
        print("   Then run this script again to verify!")

    print("=" * 60)


if __name__ == "__main__":
    main()
