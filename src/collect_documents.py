from university_document_collector import UniversityDocumentCollector
import os

def main():
    print(" University of Sargodha RAG Chatbot - Document Collection")
    print("="*60)
    
    # Initialize collector
    collector = UniversityDocumentCollector()
    
    # Collect documents
    stats = collector.collect_all_documents()
    
    # Show summary
    summary = collector.get_collection_summary()
    
    print("\n COLLECTION SUMMARY")
    print("="*40)
    print(f"PDF documents collected: {summary['pdf_count']}")
    print(f"Web pages scraped: {summary['web_count']}")
    print(f"Total PDF size: {summary['total_pdf_size_mb']} MB")
    
    print("\n PDF FILES:")
    for pdf in summary['pdf_files']:
        print(f"  - {pdf}")
    
    print("\n WEB CONTENT FILES:")
    for web in summary['web_files']:
        print(f"  - {web}")
    
    print(f"\n Files saved to:")
    print(f"  PDFs: {summary['folders']['pdfs']}")
    print(f"  Web content: {summary['folders']['web_content']}")
    
    print("\n Document collection complete!")
    print(" Please check the manual collection checklist above")
    print(" Next step: Run document processing")

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    main()