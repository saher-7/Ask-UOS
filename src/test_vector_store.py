from text_chunker import UniversityTextChunker
from embedding_service import get_embedding_service
from vector_store import UniversityVectorStore
import os
import json
import time

def main():
    print(" University Vector Store Test")
    print("="*50)
    
    # Check prerequisites
    chunks_file = "data/processed/text_chunks.json"
    if not os.path.exists(chunks_file):
        print(" No chunks found! Run text chunking first:")
        print("   python test_chunking.py")
        return
    
    # Load chunks
    print(" Loading text chunks...")
    chunker = UniversityTextChunker()
    chunks = chunker.load_chunks(chunks_file)
    
    if not chunks:
        print(" Failed to load chunks")
        return
    
    print(f" Loaded {len(chunks)} chunks")
    
    # Initialize embedding service
    print(" Initializing embedding service...")
    try:
        embedding_service, service_type = get_embedding_service()
        print(f" Using {service_type} embedding service")
    except Exception as e:
        print(f" Failed to initialize embedding service: {e}")
        return
    
    # Create embeddings for chunks
    print(" Creating embeddings for chunks...")
    chunk_texts = [chunk.content for chunk in chunks]
    
    start_time = time.time()
    embeddings = embedding_service.create_embeddings_batch(chunk_texts, show_progress=True)
    embedding_time = time.time() - start_time
    
    print(f" Created {len(embeddings)} embeddings in {embedding_time:.2f}s")
    
    # Initialize vector store
    print(" Initializing vector store...")
    vector_store = UniversityVectorStore()
    
    # Clear existing data (for clean test)
    print(" Clearing existing data...")
    vector_store.clear_collection()
    
    # Add chunks to vector store
    print(" Adding chunks to vector store...")
    start_time = time.time()
    add_results = vector_store.add_chunks(chunks, embeddings, batch_size=50)
    add_time = time.time() - start_time
    
    print(f"\n ADDITION RESULTS:")
    print(f"Successful additions: {add_results['successful_additions']}")
    print(f"Failed additions: {add_results['failed_additions']}")
    print(f"Addition time: {add_time:.2f}s")
    
    if add_results['errors']:
        print(f"Errors: {len(add_results['errors'])}")
        for error in add_results['errors'][:3]:  # Show first 3 errors
            print(f"  - {error}")
    
    # Test similarity search
    print("\n Testing similarity searches...")
    
    test_queries = [
        "What are the admission requirements for undergraduate programs?",
        "How much are the fees for postgraduate degrees?",
        "What are the rules for examinations?",
        "What are the library regulations?",
        "What disciplinary actions can be taken?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query[:50]}...")
        
        # Create query embedding
        query_embedding = embedding_service.create_embedding(query)
        
        if query_embedding:
            # Search
            results = vector_store.similarity_search(query_embedding, n_results=3)
            
            print(f"Found {results['total_found']} results:")
            
            for j, (doc, metadata, distance, score) in enumerate(zip(
                results['documents'][:2],  # Show top 2
                results['metadatas'][:2], 
                results['distances'][:2],
                results['scores'][:2] if results['scores'] else [0, 0]
            )):
                print(f"  Result {j+1}:")
                print(f"    Category: {metadata.get('document_category', 'unknown')}")
                print(f"    Section: {metadata.get('section_title', 'N/A')}")
                print(f"    Score: {score:.3f}")
                print(f"    Content: {doc[:100]}...")
        else:
            print("  Failed to create query embedding")
    
    # Test category search
    print("\n Testing category search...")
    query_embedding = embedding_service.create_embedding("admission process")
    
    if query_embedding:
        admission_results = vector_store.search_by_category(query_embedding, "admission", n_results=2)
        print(f"Found {admission_results['total_found']} admission-related results")
        
        for doc, metadata in zip(admission_results['documents'], admission_results['metadatas']):
            print(f"  - {metadata.get('section_title', 'N/A')}: {doc[:80]}...")
    
    # Test rules-only search
    print("\n Testing rules-only search...")
    if query_embedding:
        rules_results = vector_store.search_rules_only(query_embedding, n_results=2)
        print(f"Found {rules_results['total_found']} rule-related results")
    
    # Get and display statistics
    print("\n VECTOR STORE STATISTICS")
    print("="*40)
    
    stats = vector_store.get_collection_stats()
    
    if 'error' not in stats:
        print(f"Total documents: {stats['total_documents']}")
        print(f"Sample size: {stats['sample_size']}")
        
        if 'averages' in stats:
            print(f"\nAverages:")
            for key, value in stats['averages'].items():
                print(f"  {key}: {value}")

        print(f"\nEmbedding dimension: {stats.get('embedding_dimension', 'N/A')}")

        print(f"\nCategory distribution:")
        for category, count in stats['distributions']['categories'].items():
            print(f"  {category}: {count}")

        print(f"\nContent analysis:")
        for content_type, info in stats['content_analysis'].items():
            print(f"  {content_type}: {info}")

        print(f"\nSearch statistics:")
        for key, value in stats['search_stats'].items():
            print(f"  {key}: {value}")
    else:
        print(f"Error getting stats: {stats['error']}")
    
    # Export collection
    print("\n Exporting collection...")
    if vector_store.export_collection():
        print(" Collection exported successfully")
    
    # Show embedding service stats
    print("\n EMBEDDING SERVICE STATISTICS")
    print("="*40)
    
    service_stats = embedding_service.get_stats()
    for key, value in service_stats.items():
        print(f"{key}: {value}")
    
    print("\n Vector store test complete!")
    print(" Next step: Build RAG pipeline")

if __name__ == "__main__":
    main()