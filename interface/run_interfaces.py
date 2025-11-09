#!/usr/bin/env python3
"""
Launcher script for University of Sargodha RAG Chatbot interfaces
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Check if system is ready to run interfaces"""
    print(" Checking system requirements...")
    
    # Check if vector store has data
    vector_db_path = Path("models/faiss_db")
    if not vector_db_path.exists() or not list(vector_db_path.iterdir()):
        print(" Vector database not found!")
        print("Please run the setup process first:")
        print("  python collect_documents.py")
        print("  python test_processing.py")
        print("  python test_chunking.py") 
        print("  python test_vector_store.py")
        return False
    
    # Check if required packages are installed
    try:
        import streamlit
        import gradio
        print(" Required packages found")
    except ImportError as e:
        print(f" Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False
    
    print(" System ready to run interfaces")
    return True

def run_streamlit():
    """Run Streamlit interface"""
    print(" Starting Streamlit interface...")
    print(" Web interface will open at: http://localhost:8501")
    print("  Press Ctrl+C to stop")
    print("-" * 50)
    
    cmd = [sys.executable, "-m", "streamlit", "run", "interface/streamlit_app.py"]
    subprocess.run(cmd)

def run_gradio():
    """Run Gradio interface"""
    print(" Starting Gradio interface...")
    print(" Web interface will open at: http://localhost:7860")
    print("  Press Ctrl+C to stop")
    print("-" * 50)
    
    cmd = [sys.executable, "interface/gradio_app.py"]
    subprocess.run(cmd)

def run_cli():
    """Run CLI interface"""
    print(" Starting CLI interface...")
    print(" Interactive chat mode")
    print("  Type 'quit' to exit")
    print("-" * 50)
    
    cmd = [sys.executable, "interface/cli_app.py", "--interactive"]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(
        description="University of Sargodha RAG Chatbot Interface Launcher"
    )
    
    parser.add_argument(
        'interface',
        choices=['streamlit', 'gradio', 'cli', 'all'],
        help='Interface to run'
    )
    
    parser.add_argument(
        '--skip-check',
        action='store_true',
        help='Skip system requirements check'
    )
    
    args = parser.parse_args()
    
    print(" University of Sargodha RAG Chatbot")
    print("="*50)
    
    # Check requirements unless skipped
    if not args.skip_check and not check_requirements():
        sys.exit(1)
    
    # Run selected interface
    if args.interface == 'streamlit':
        run_streamlit()
    elif args.interface == 'gradio':
        run_gradio()
    elif args.interface == 'cli':
        run_cli()
    elif args.interface == 'all':
        print(" Available interfaces:")
        print("1. Streamlit: python run_interfaces.py streamlit")
        print("2. Gradio: python run_interfaces.py gradio") 
        print("3. CLI: python run_interfaces.py cli")
        print("\nPlease select one interface to run.")

if __name__ == "__main__":
    main()
