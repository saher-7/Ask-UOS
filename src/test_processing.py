from document_processor import UniversityDocumentProcessor
import os
import json

def main():
    print(" University of Sargodha - Document Processing Test")
    print("="*60)
    
    # Initialize processor
    processor = UniversityDocumentProcessor()
    
    # Check if documents exist
    pdf_folder = "data/pdfs"
    web_folder = "data/web_content"
    
    pdf_count = len([f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]) if os.path.exists(pdf_folder) else 0
    txt_count = len([f for f in os.listdir(web_folder) if f.endswith('.txt')]) if os.path.exists(web_folder) else 0
    
    if pdf_count == 0 and txt_count == 0:
        print(" No documents found!")
        print(" Please run document collection first:")
        print("   python collect_documents.py")
        return
    
    print(f" Found {pdf_count} PDF files and {txt_count} text files")
    
    # Process all documents
    results = processor.process_all_documents()
    
    # Display results
    print("\n PROCESSING RESULTS")
    print("="*40)
    
    stats = results['statistics']
    print(f"Total files: {stats['total_files']}")
    print(f"Processed: {stats['processed_files']}")
    print(f"Failed: {stats['failed_files']}")
    print(f"Total characters: {stats['total_characters']:,}")
    
    if stats['processing_errors']:
        print(f"\n Errors ({len(stats['processing_errors'])}):")
        for error in stats['processing_errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    # Show sample of combined text
    combined_text = results['combined_text']
    if combined_text:
        print(f"\n Combined document: {len(combined_text):,} characters")
        print(" First 500 characters:")
        print("-" * 40)
        print(combined_text[:500] + "...")
        print("-" * 40)
    
    # Show file locations
    summary = processor.get_processing_summary()
    print(f"\n FILES SAVED:")
    for location_type, path in summary['files_locations'].items():
        print(f"  {location_type}: {path}")
    
    print("\n Document processing complete!")
    print(" Next step: Run text chunking")

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    main()