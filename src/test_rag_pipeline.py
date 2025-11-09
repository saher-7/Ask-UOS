from rag_pipeline import UniversityRAGPipeline
from vector_store import UniversityVectorStore
from embedding_service import get_embedding_service
import time
import json

def test_basic_rag_functionality():
    print(" Testing Basic RAG Functionality")
    print("="*50)
    
    try:
        # Initialize RAG pipeline
        print(" Initializing RAG pipeline...")
        rag = UniversityRAGPipeline()
        
        # Test queries for different categories
        test_queries = [
            "What are the admission requirements for undergraduate programs?",
            "How much are the fees for MS programs?",
            "What are the examination rules and procedures?",
            "What are the library borrowing rules?",
            "What disciplinary actions can be taken for misconduct?",
            "Tell me about the hostel accommodation rules",
            "What are the requirements for PhD admission?",
            "How do I apply for semester registration?",
            "What is the fee refund policy?",
            "What are the attendance requirements?"
        ]
        
        print(f" Testing {len(test_queries)} queries...")
        
        all_responses = []
        total_time = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}/{len(test_queries)} ---")
            print(f"Q: {query}")
            
            start_time = time.time()
            response = rag.process_query(query)
            query_time = time.time() - start_time
            
            total_time += query_time
            
            print(f"  Response time: {response.response_time:.2f}s")
            print(f" Confidence: {max(response.confidence_scores) if response.confidence_scores else 0:.3f}")
            print(f" Sources used: {response.sources_used}")
            print(f" Tokens used: {response.total_tokens_used}")
            
            # Show answer (truncated)
            answer_preview = response.answer[:300] + "..." if len(response.answer) > 300 else response.answer
            print(f"A: {answer_preview}")
            
            # Show top source (if available)
            if response.source_chunks and response.metadata:
                source_preview = response.source_chunks[0][:150] + "..."
                source_title = response.metadata[0].get('section_title', 'Unknown')
                print(f" Top source: {source_title}")
                print(f"   Content: {source_preview}")
            
            all_responses.append({
                'query': query,
                'response_time': response.response_time,
                'confidence': max(response.confidence_scores) if response.confidence_scores else 0,
                'sources_used': response.sources_used,
                'tokens_used': response.total_tokens_used,
                'answer_length': len(response.answer)
            })
        
        # Summary statistics
        print(f"\n TEST SUMMARY")
        print("="*30)
        print(f"Total queries: {len(test_queries)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per query: {total_time/len(test_queries):.2f}s")
        
        avg_confidence = sum(r['confidence'] for r in all_responses) / len(all_responses)
        avg_sources = sum(r['sources_used'] for r in all_responses) / len(all_responses)
        total_tokens = sum(r['tokens_used'] for r in all_responses)
        
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Average sources per query: {avg_sources:.1f}")
        print(f"Total tokens used: {total_tokens}")
        
        # Check for responses with low confidence
        low_confidence = [r for r in all_responses if r['confidence'] < 0.3]
        if low_confidence:
            print(f"  Low confidence responses: {len(low_confidence)}")
            for resp in low_confidence:
                print(f"  - {resp['query'][:50]}... (confidence: {resp['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f" RAG test failed: {e}")
        return False

def test_conversation_context():
    print("\n Testing Conversation Context")
    print("="*40)
    
    try:
        rag = UniversityRAGPipeline()
        
        # Simulate conversation
        conversation_history = []
        
        queries = [
            "What are the admission requirements for MS programs?",
            "What about the fees for these programs?",
            "Are there any scholarships available?",
            "How do I apply for them?"
        ]
        
        for i, query in enumerate(queries):
            print(f"\nTurn {i+1}: {query}")
            
            # Use conversation context for follow-up questions
            if conversation_history:
                context_query = rag.get_conversation_context(query, conversation_history)
                print(f"Context-aware query: {context_query[:100]}...")
                response = rag.process_query(context_query)
            else:
                response = rag.process_query(query)
            
            print(f"Response: {response.answer[:200]}...")
            
            # Add to conversation history
            conversation_history.append({
                'user': query,
                'assistant': response.answer
            })
        
        return True
        
    except Exception as e:
        print(f" Conversation context test failed: {e}")
        return False

def test_category_specific_queries():
    print("\n Testing Category-Specific Queries")
    print("="*40)
    
    try:
        rag = UniversityRAGPipeline()
        
        category_queries = {
            'admission': [
                "What documents are required for admission?",
                "What is the merit calculation process?",
                "When is the admission deadline?"
            ],
            'fees': [
                "What is the fee structure for different programs?",
                "Are there any additional charges?",
                "What is the fee payment schedule?"
            ],
            'examination': [
                "What are the examination rules?",
                "How are grades calculated?",
                "What happens if I miss an exam?"
            ]
        }
        
        for category, queries in category_queries.items():
            print(f"\n--- {category.upper()} QUERIES ---")
            
            for query in queries:
                print(f"Q: {query}")
                response = rag.process_query(query)
                
                # Check if response contains category-relevant information
                answer_lower = response.answer.lower()
                category_relevant = any(word in answer_lower for word in [category, category[:-1]])
                
                print(f" Category relevant: {category_relevant}")
                print(f" Sources: {response.sources_used}")
                print(f"A: {response.answer[:150]}...\n")
        
        return True
        
    except Exception as e:
        print(f" Category-specific test failed: {e}")
        return False

def test_edge_cases():
    print("\n Testing Edge Cases")
    print("="*30)
    
    try:
        rag = UniversityRAGPipeline()
        
        edge_cases = [
            "",  # Empty query
            "hello",  # Irrelevant query
            "What is the meaning of life?",  # Philosophical query
            "Tell me about University of Punjab",  # Wrong university
            "a" * 1000,  # Very long query
            "admission fee examination library hostel",  # Multiple categories
        ]
        
        for i, query in enumerate(edge_cases, 1):
            query_preview = query if len(query) <= 50 else query[:50] + "..."
            print(f"\nEdge case {i}: '{query_preview}'")
            
            try:
                response = rag.process_query(query)
                print(f" Handled successfully")
                print(f"Response length: {len(response.answer)}")
                print(f"Sources used: {response.sources_used}")
                
                # Check if response is appropriate
                if not query.strip():
                    print("Empty query handled appropriately" if "question" in response.answer.lower() else "  Unexpected response to empty query")
                
            except Exception as e:
                print(f" Failed to handle edge case: {e}")
        
        return True
        
    except Exception as e:
        print(f" Edge case testing failed: {e}")
        return False

def main():
    print(" University RAG Pipeline - Comprehensive Test Suite")
    print("="*70)
    
    # Check if vector store has data
    try:
        vector_store = UniversityVectorStore()
        stats = vector_store.get_collection_stats()
        
        if stats.get('total_documents', 0) == 0:
            print(" Vector store is empty!")
            print(" Please run the vector store setup first:")
            print("   python test_vector_store.py")
            return
        
        print(f" Vector store ready with {stats['total_documents']} documents")
        
    except Exception as e:
        print(f" Vector store check failed: {e}")
        return
    
    # Run all tests
    tests = [
        ("Basic RAG Functionality", test_basic_rag_functionality),
        ("Conversation Context", test_conversation_context),
        ("Category-Specific Queries", test_category_specific_queries),
        ("Edge Cases", test_edge_cases)
    ]
    
    passed = 0
    
    for test_name, test_function in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_function():
                print(f" {test_name}: PASSED")
                passed += 1
            else:
                print(f" {test_name}: FAILED")
        except Exception as e:
            print(f" {test_name}: ERROR - {e}")
    
    # Final results
    print(f"\n{'='*70}")
    print(f" TEST RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print(" All tests passed! RAG pipeline is ready for deployment.")
    else:
        print("  Some tests failed. Please review the issues above.")
    
    # Show pipeline statistics
    try:
        rag = UniversityRAGPipeline()
        stats = rag.get_statistics()
        
        print(f"\n PIPELINE STATISTICS")
        print("="*30)
        
        for section, section_stats in stats.items():
            if isinstance(section_stats, dict):
                print(f"\n{section.replace('_', ' ').title()}:")
                for key, value in section_stats.items():
                    print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"  Could not retrieve statistics: {e}")
    
    print("\n RAG Pipeline testing complete!")
    print(" Next step: Create chatbot interface")

if __name__ == "__main__":
    main()