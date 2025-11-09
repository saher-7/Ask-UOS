#!/usr/bin/env python3
"""
Command Line Interface for University of Sargodha RAG Chatbot
"""

import sys
import os
import time
from datetime import datetime
import argparse
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import UniversityRAGPipeline
from src.vector_store import UniversityVectorStore

class UniversityChatCLI:
    def __init__(self):
        self.rag_pipeline = None
        self.conversation_history = []
        self.session_start = datetime.now()
        
    def initialize_system(self):
        """Initialize the RAG pipeline system"""
        print(" Initializing University of Sargodha Chatbot...")
        print("-" * 50)
        
        try:
            # Check vector store
            print(" Checking knowledge base...")
            vector_store = UniversityVectorStore()
            stats = vector_store.get_collection_stats()
            
            if stats.get('total_documents', 0) == 0:
                print(" Error: No documents found in vector store.")
                print("Please run the setup process first:")
                print("  python collect_documents.py")
                print("  python test_processing.py") 
                print("  python test_chunking.py")
                print("  python test_vector_store.py")
                return False
            
            print(f" Knowledge base loaded: {stats['total_documents']} documents")
            
            # Initialize pipeline
            print(" Initializing RAG pipeline...")
            self.rag_pipeline = UniversityRAGPipeline(vector_store=vector_store)
            
            print(" System ready!")
            print("-" * 50)
            
            return True
            
        except Exception as e:
            print(f" Failed to initialize system: {e}")
            return False
    
    def display_welcome(self):
        """Display welcome message"""
        print("\n" + "=" * 60)
        print(" University of Sargodha - Rules & Regulations Chatbot")
        print("=" * 60)
        print()
        print("Welcome! I can help you find information about:")
        print(" Admission requirements and procedures")
        print(" Fee structures and payment information")
        print(" Examination rules and grading policies")
        print(" Hostel and accommodation guidelines")
        print(" Library rules and regulations")
        print("  Disciplinary procedures and policies")
        print()
        print(" Tips:")
        print("- Type 'help' for available commands")
        print("- Type 'examples' for sample questions")
        print("- Type 'quit' or 'exit' to end session")
        print("-" * 60)
    
    def display_help(self):
        """Display help information"""
        print("\n Available Commands:")
        print("-" * 30)
        print("help          - Show this help message")
        print("examples      - Show sample questions")
        print("stats         - Show system statistics")
        print("history       - Show conversation history")
        print("clear         - Clear conversation history")
        print("save          - Save conversation to file")
        print("quit/exit     - Exit the chatbot")
        print("-" * 30)
    
    def display_examples(self):
        """Display sample questions"""
        examples = [
            "What are the admission requirements for undergraduate programs?",
            "How much are the fees for Master's degree programs?",
            "What are the examination rules and procedures?",
            "What documents are required for admission?",
            "What are the library borrowing rules?",
            "What disciplinary actions can be taken against students?",
            "How can I apply for a postgraduate program?",
            "What is the semester system at University of Sargodha?",
            "What are the hostel accommodation rules?",
            "How are examination results calculated?"
        ]
        
        print("\n Sample Questions:")
        print("-" * 40)
        for i, example in enumerate(examples, 1):
            print(f"{i:2d}. {example}")
        print("-" * 40)
    
    def display_stats(self):
        """Display system statistics"""
        if not self.rag_pipeline:
            print(" System not initialized")
            return
        
        try:
            stats = self.rag_pipeline.get_pipeline_stats()
            session_time = datetime.now() - self.session_start
            
            print("\n System Statistics:")
            print("=" * 40)
            print(f"Session duration: {session_time}")
            print(f"Total queries: {stats['queries']['total_queries']}")
            print(f"Successful responses: {stats['queries']['successful_responses']}")
            print(f"Success rate: {stats['queries']['success_rate']}%")
            print(f"Average response time: {stats['performance']['average_response_time']:.2f}s")
            print(f"Total tokens used: {stats['performance']['total_tokens_used']:,}")
            
            if stats.get('vector_store'):
                vs_stats = stats['vector_store']
                print(f"Knowledge base size: {vs_stats.get('total_documents', 'N/A')}")
            
            print("=" * 40)
            
        except Exception as e:
            print(f" Error getting statistics: {e}")
    
    def display_history(self):
        """Display conversation history"""
        if not self.conversation_history:
            print(" No conversation history yet.")
            return
        
        print(f"\n Conversation History ({len(self.conversation_history)} interactions):")
        print("-" * 60)
        
        for i, interaction in enumerate(self.conversation_history, 1):
            timestamp = interaction.get('timestamp', 'Unknown time')
            query = interaction['query']
            response_time = interaction.get('response_time', 0)
            sources_used = interaction.get('sources_used', 0)
            
            print(f"\n{i}. [{timestamp}]")
            print(f" Q: {query}")
            print(f" Info: {response_time:.2f}s, {sources_used} sources")
            print(f" A: {interaction['answer'][:200]}...")
            print("-" * 60)
    
    def save_conversation(self):
        """Save conversation history to file"""
        if not self.conversation_history:
            print(" No conversation to save.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"su_chatbot_session_{timestamp}.json"
        
        try:
            conversation_data = {
                "session_info": {
                    "start_time": self.session_start.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_interactions": len(self.conversation_history)
                },
                "university": "University of Sargodha",
                "conversation": self.conversation_history
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            print(f" Conversation saved to: {filename}")
            
        except Exception as e:
            print(f" Error saving conversation: {e}")
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("  Conversation history cleared.")
    
    def process_query(self, query):
        """Process user query and return response"""
        if not self.rag_pipeline:
            return " System not initialized"
        
        print(f"\n Processing: {query}")
        print(" Please wait...")
        
        start_time = time.time()
        
        try:
            response = self.rag_pipeline.query(query)
            processing_time = time.time() - start_time
            
            # Display response
            print(f"\n University Assistant:")
            print("-" * 50)
            print(response.answer)
            print("-" * 50)
            
            # Display metadata
            print(f" Response Info:")
            print(f"     Response time: {response.response_time:.2f}s (total: {processing_time:.2f}s)")
            print(f"    Sources used: {response.sources_used}")
            print(f"    Confidence: {sum(response.confidence_scores)/len(response.confidence_scores)*100:.1f}%" if response.confidence_scores else "    Confidence: N/A")
            print(f"    Tokens used: {response.total_tokens_used}")
            
            # Save to history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "answer": response.answer,
                "response_time": response.response_time,
                "sources_used": response.sources_used,
                "confidence_scores": response.confidence_scores,
                "tokens_used": response.total_tokens_used
            })
            
            return response.answer
            
        except Exception as e:
            error_msg = f" Error processing query: {e}"
            print(error_msg)
            return error_msg
    
    def run_interactive(self):
        """Run interactive chat session"""
        if not self.initialize_system():
            return
        
        self.display_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("\n You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n Thank you for using University of Sargodha Chatbot!")
                    if self.conversation_history:
                        save_choice = input(" Save conversation? (y/n): ").strip().lower()
                        if save_choice in ['y', 'yes']:
                            self.save_conversation()
                    break
                
                elif user_input.lower() == 'help':
                    self.display_help()
                    
                elif user_input.lower() == 'examples':
                    self.display_examples()
                    
                elif user_input.lower() == 'stats':
                    self.display_stats()
                    
                elif user_input.lower() == 'history':
                    self.display_history()
                    
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    
                elif user_input.lower() == 'save':
                    self.save_conversation()
                
                else:
                    # Process as query
                    self.process_query(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n Session interrupted. Goodbye!")
                break
            except EOFError:
                print("\n\n Session ended. Goodbye!")
                break
            except Exception as e:
                print(f"\n Unexpected error: {e}")
                print("Please try again or type 'quit' to exit.")
    
    def run_single_query(self, query):
        """Run single query mode"""
        if not self.initialize_system():
            return
        
        response = self.process_query(query)
        return response

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="University of Sargodha Rules & Regulations Chatbot CLI"
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help="Single query mode - ask one question and exit"
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help="Interactive chat mode (default)"
    )
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = UniversityChatCLI()
    
    if args.query:
        # Single query mode
        cli.run_single_query(args.query)
    else:
        # Interactive mode (default)
        cli.run_interactive()

if __name__ == "__main__":
    main()