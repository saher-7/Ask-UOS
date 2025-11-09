from document_processor import UniversityDocumentProcessor
from text_chunker import UniversityTextChunker
import os

def main():
    print(" University of Sargodha - Text Chunking Test")
    print("="*60)
    
    # Check if processed document exists
    combined_doc_path = "data/processed/university_rules_combined.txt"
    
    if not os.path.exists(combined_doc_path):
        print(" Combined document not found!")
        print(" Please run document processing first:")
        print("   python test_processing.py")
        return
    
    # Load combined document
    with open(combined_doc_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f" Loaded document: {len(text):,} characters")
    
    # Initialize chunker
    chunker = UniversityTextChunker(
        chunk_size=2000,
        chunk_overlap=200,
        model_name="gpt-3.5-turbo"
    )
    
    # Create chunks
    print("\n Creating chunks...")
    chunks = chunker.create_chunks(text, source_document="university_rules_combined.txt")
    
    # Save chunks
    chunker.save_chunks(chunks)
    
    # Display results
    print("\n CHUNKING RESULTS")
    print("="*40)
    
    summary = chunker.get_chunking_summary()
    stats = summary['statistics']
    
    print(f"Total chunks created: {stats['total_chunks']}")
    print(f"Original text length: {stats['total_characters']:,} characters")
    print(f"Average chunk size: {stats['average_chunk_size']} characters")
    print(f"Average tokens per chunk: {stats['average_tokens_per_chunk']}")
    print(f"Processing time: {stats['processing_time']:.2f} seconds")
    
    print(f"\n Chunk size distribution:")
    for size_range, count in stats['chunk_size_distribution'].items():
        percentage = (count / stats['total_chunks']) * 100
        print(f"  {size_range} chars: {count} chunks ({percentage:.1f}%)")
    
    # Show sample chunks
    print(f"\n SAMPLE CHUNKS (First 3):")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {chunk.chunk_id} ---")
        print(f"Source: {chunk.source_document}")
        print(f"Section: {chunk.section_title}")
        print(f"Type: {chunk.chunk_type}")
        print(f"Size: {len(chunk.content)} chars, {chunk.token_count} tokens")
        print(f"Content: {chunk.content[:200]}...")
    
    # Show files created
    print(f"\n FILES CREATED:")
    for file_type, path in summary['files'].items():
        print(f"  {file_type}: {path}")
    
    print("\n Text chunking complete!")
    print(" Next step: Set up embeddings")

if __name__ == "__main__":
    main()