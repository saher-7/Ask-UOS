from embedding_service import get_embedding_service, EmbeddingService, LocalEmbeddingService
from text_chunker import UniversityTextChunker
import os
import json
from dotenv import load_dotenv

def test_embedding_services():
    print(" Testing Embedding Services")
    print("="*50)
    
    # Test texts
    test_texts = [
        "University of Sargodha admission requirements for undergraduate programs.",
        "Rules and regulations for examination procedures.",
        "Fee structure for postgraduate degree programs.",
        "Library rules and borrowing procedures for students.",
        "Disciplinary actions and academic misconduct policies."
    ]
    
    # Test OpenAI service (if available)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("\n Testing OpenAI Embedding Service...")
        try:
            openai_service = EmbeddingService()
            
            if openai_service.test_connection():
                # Test single embedding
                single_embedding = openai_service.create_embedding(test_texts[0])
                print(f" Single embedding: {len(single_embedding)} dimensions")
                
                # Test batch embeddings
                batch_embeddings = openai_service.create_embeddings_batch(test_texts[:3])
                print(f" Batch embeddings: {len(batch_embeddings)} embeddings")
                
                # Show statistics
                stats = openai_service.get_stats()
                print(f" OpenAI Stats:")
                print(f"  Requests: {stats['total_requests']}")
                print(f"  Cache hits: {stats['cache_hits']} ({stats['cache_hit_rate']}%)")
                print(f"  API calls: {stats['api_calls']}")
                print(f"  Estimated cost: ${stats['estimated_cost_usd']}")
            
        except Exception as e:
            print(f" OpenAI service error: {e}")
    else:
        print("  No OpenAI API key found, skipping OpenAI test")
    
    # Test Local service
    print("\n Testing Local Embedding Service...")
    try:
        local_service = LocalEmbeddingService()
        
        if local_service.test_connection():
            # Test single embedding
            single_embedding = local_service.create_embedding(test_texts[0])
            print(f" Single embedding: {len(single_embedding)} dimensions")
            
            # Test batch embeddings
            batch_embeddings = local_service.create_embeddings_batch(test_texts)
            print(f" Batch embeddings: {len(batch_embeddings)} embeddings")
            
            # Show statistics
            stats = local_service.get_stats()
            print(f" Local Stats:")
            print(f"  Requests: {stats['total_requests']}")
            print(f"  Cache hits: {stats['cache_hits']} ({stats['cache_hit_rate']}%)")
            print(f"  Model calls: {stats['model_calls']}")
            print(f"  Processing time: {stats['total_processing_time']}s")
        
    except Exception as e:
        print(f" Local service error: {e}")
    
    # Test auto-detection
    print("\n Testing Auto-Detection...")
    try:
        service, service_type = get_embedding_service()
        print(f" Auto-selected: {service_type} service")
        
        test_embedding = service.create_embedding("Auto-detection test")
        print(f" Auto-service working: {len(test_embedding)} dimensions")
        
    except Exception as e:
        print(f" Auto-detection error: {e}")


def test_chunk_embeddings():
    print("\n Testing Chunk Embeddings")
    print("="*40)
    
    # Check if chunks exist
    chunks_file = "data/processed/text_chunks.json"
    if not os.path.exists(chunks_file):
        print(" No chunks found! Run text chunking first.")
        return
    
    # Load chunks
    chunker = UniversityTextChunker()
    chunks = chunker.load_chunks(chunks_file)
    
    if not chunks:
        print(" Failed to load chunks")
        return
    
    print(f" Loaded {len(chunks)} chunks")
    
    # Get embedding service
    try:
        service, service_type = get_embedding_service()
        print(f" Using {service_type} embedding service")
    except Exception as e:
        print(f" Failed to initialize embedding service: {e}")
        return
    
    # Test with first few chunks
    test_chunks = chunks[:5]
    chunk_texts = [chunk.content for chunk in test_chunks]
    
    print(f" Creating embeddings for {len(test_chunks)} sample chunks...")
    
    try:
        embeddings = service.create_embeddings_batch(chunk_texts)
        
        print(f" Created {len(embeddings)} embeddings")
        
        # Show embedding info
        for i, (chunk, embedding) in enumerate(zip(test_chunks, embeddings)):
            print(f"\nChunk {i+1}:")
            print(f"  Content length: {len(chunk.content)} chars")
            print(f"  Embedding dimensions: {len(embedding)}")
            print(f"  Sample values: {embedding[:5]}")
        
        # Show service stats
        stats = service.get_stats()
        print(f"\n Service Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f" Error creating embeddings: {e}")


def main():
    load_dotenv()
    
    print(" Embedding Service Test Suite")
    print("="*60)
    
    # Test basic services
    test_embedding_services()
    
    # Test with actual chunks
    test_chunk_embeddings()
    
    print("\n Embedding tests complete!")
    print(" Next step: Set up vector database")


if __name__ == "__main__":
    main()