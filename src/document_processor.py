import os
import PyPDF2
from docx import Document
from typing import List, Dict, Tuple, Optional
import logging
import json
from pathlib import Path
import re
from datetime import datetime

class UniversityDocumentProcessor:
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.pdf_folder = os.path.join(data_folder, "pdfs")
        self.web_folder = os.path.join(data_folder, "web_content")
        self.processed_folder = os.path.join(data_folder, "processed")
        
        # Create processed folder
        os.makedirs(self.processed_folder, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Supported formats
        self.supported_formats = ['.pdf', '.txt', '.docx']
        
        # Processing statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_characters': 0,
            'processing_errors': []
        }
    
    def validate_document(self, file_path: str) -> Tuple[bool, str]:
        """Validate document with detailed error reporting"""
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        if os.path.getsize(file_path) == 0:
            return False, f"File is empty: {file_path}"
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_formats:
            return False, f"Unsupported format: {file_ext}. Supported: {self.supported_formats}"
        
        # Check file permissions
        if not os.access(file_path, os.R_OK):
            return False, f"File not readable: {file_path}"
        
        return True, "Valid"
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extract text from PDF with metadata"""
        metadata = {
            'source': pdf_path,
            'type': 'PDF',
            'pages_processed': 0,
            'pages_failed': 0,
            'extraction_method': 'PyPDF2'
        }
        
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                metadata['total_pages'] = total_pages
                
                self.logger.info(f" Processing PDF: {os.path.basename(pdf_path)} ({total_pages} pages)")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        
                        if page_text.strip():
                            # Clean page text
                            page_text = self.clean_pdf_text(page_text)
                            
                            # Add page marker
                            text += f"\n\n--- PAGE {page_num + 1} ---\n\n"
                            text += page_text
                            metadata['pages_processed'] += 1
                        
                        # Progress indicator
                        if (page_num + 1) % 10 == 0 or page_num == total_pages - 1:
                            progress = ((page_num + 1) / total_pages) * 100
                            print(f"\r Progress: {progress:.1f}% ({page_num + 1}/{total_pages})", end="")
                    
                    except Exception as e:
                        self.logger.warning(f"  Failed to extract page {page_num + 1}: {e}")
                        metadata['pages_failed'] += 1
                        continue
                
                print()  # New line after progress
            
            metadata['character_count'] = len(text)
            self.logger.info(f" PDF processed: {metadata['pages_processed']}/{total_pages} pages, {len(text)} chars")
            return text, metadata
            
        except Exception as e:
            error_msg = f"Error reading PDF {pdf_path}: {e}"
            self.logger.error(f" {error_msg}")
            metadata['error'] = error_msg
            return "", metadata
    
    def clean_pdf_text(self, text: str) -> str:
        """Enhanced PDF text cleaning"""
        if not text:
            return ""
        
        # Fix common PDF extraction issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words across lines
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'Page \d+.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'University of Sargodha.*?\n', '', text, flags=re.IGNORECASE)
        
        # Fix common OCR errors
        ocr_fixes = {
            '|': 'I',
            '0': 'O',  # Context-dependent, be careful
            '©': '(c)',
            '®': '(R)',
            '™': '(TM)'
        }
        
        for wrong, right in ocr_fixes.items():
            # Only fix if it makes sense in context
            if wrong in text:
                text = text.replace(wrong, right)
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])\s+', r'\1 ', text)
        
        return text.strip()
    
    def extract_text_from_txt(self, txt_path: str) -> Tuple[str, Dict]:
        """Extract text from TXT file with metadata"""
        metadata = {
            'source': txt_path,
            'type': 'TXT',
            'encoding_used': 'utf-8'
        }
        
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(txt_path, 'r', encoding=encoding) as file:
                    text = file.read()
                
                metadata['encoding_used'] = encoding
                metadata['character_count'] = len(text)
                
                # Clean text
                text = self.clean_txt_text(text)
                
                self.logger.info(f" TXT processed: {os.path.basename(txt_path)} ({len(text)} chars)")
                return text, metadata
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                error_msg = f"Error reading TXT {txt_path}: {e}"
                self.logger.error(f" {error_msg}")
                metadata['error'] = error_msg
                return "", metadata
        
        error_msg = f"Could not decode TXT file with any encoding: {txt_path}"
        self.logger.error(f" {error_msg}")
        metadata['error'] = error_msg
        return "", metadata
    
    def clean_txt_text(self, text: str) -> str:
        """Clean text from TXT files"""
        if not text:
            return ""
        
        # Remove URL metadata if present
        if text.startswith("Source URL:"):
            lines = text.split('\n')
            # Skip first few metadata lines
            content_start = 0
            for i, line in enumerate(lines):
                if '=' in line and len(line) > 20:  # Separator line
                    content_start = i + 1
                    break
            text = '\n'.join(lines[content_start:])
        
        # Basic cleaning
        text = re.sub(r'\r\n', '\n', text)  # Windows line endings
        text = re.sub(r'\r', '\n', text)    # Mac line endings
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'[ \t]+', ' ', text)     # Multiple spaces/tabs to single space
        
        return text.strip()
    
    def extract_text_from_docx(self, docx_path: str) -> Tuple[str, Dict]:
        """Extract text from DOCX file with metadata"""
        metadata = {
            'source': docx_path,
            'type': 'DOCX',
            'paragraphs_processed': 0
        }
        
        try:
            doc = Document(docx_path)
            text = ""
            
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
                    metadata['paragraphs_processed'] += 1
            
            metadata['character_count'] = len(text)
            
            # Clean text
            text = self.clean_txt_text(text)  # Same cleaning as TXT
            
            self.logger.info(f" DOCX processed: {os.path.basename(docx_path)} ({len(text)} chars)")
            return text, metadata
            
        except Exception as e:
            error_msg = f"Error reading DOCX {docx_path}: {e}"
            self.logger.error(f" {error_msg}")
            metadata['error'] = error_msg
            return "", metadata
    
    def process_single_document(self, file_path: str) -> Tuple[str, Dict]:
        """Process a single document and return text with metadata"""
        # Validate document
        is_valid, validation_msg = self.validate_document(file_path)
        if not is_valid:
            self.logger.error(f" Validation failed: {validation_msg}")
            return "", {'error': validation_msg}
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Process based on file type
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_path)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        else:
            error_msg = f"Unsupported file type: {file_ext}"
            return "", {'error': error_msg}
    
    def process_all_documents(self) -> Dict[str, Dict]:
        """Process all documents in the data folders"""
        self.logger.info(" Starting document processing...")
        
        all_documents = {}
        all_metadata = {}
        
        # Reset statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_characters': 0,
            'processing_errors': []
        }
        
        # Process PDF files
        if os.path.exists(self.pdf_folder):
            pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
            self.logger.info(f" Found {len(pdf_files)} PDF files")
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(self.pdf_folder, pdf_file)
                self.stats['total_files'] += 1
                
                text, metadata = self.process_single_document(pdf_path)
                
                if text and 'error' not in metadata:
                    document_key = f"PDF_{pdf_file}"
                    all_documents[document_key] = text
                    all_metadata[document_key] = metadata
                    self.stats['processed_files'] += 1
                    self.stats['total_characters'] += len(text)
                    self.logger.info(f" Processed: {pdf_file}")
                else:
                    self.stats['failed_files'] += 1
                    error_msg = metadata.get('error', 'Unknown error')
                    self.stats['processing_errors'].append(f"{pdf_file}: {error_msg}")
                    self.logger.error(f" Failed: {pdf_file}")
        
        # Process TXT files (web content)
        if os.path.exists(self.web_folder):
            txt_files = [f for f in os.listdir(self.web_folder) if f.endswith('.txt')]
            self.logger.info(f" Found {len(txt_files)} TXT files")
            
            for txt_file in txt_files:
                txt_path = os.path.join(self.web_folder, txt_file)
                self.stats['total_files'] += 1
                
                text, metadata = self.process_single_document(txt_path)
                
                if text and 'error' not in metadata:
                    document_key = f"WEB_{txt_file}"
                    all_documents[document_key] = text
                    all_metadata[document_key] = metadata
                    self.stats['processed_files'] += 1
                    self.stats['total_characters'] += len(text)
                    self.logger.info(f" Processed: {txt_file}")
                else:
                    self.stats['failed_files'] += 1
                    error_msg = metadata.get('error', 'Unknown error')
                    self.stats['processing_errors'].append(f"{txt_file}: {error_msg}")
                    self.logger.error(f" Failed: {txt_file}")
        
        # Process DOCX files (if any)
        docx_files = []
        for folder in [self.pdf_folder, self.web_folder]:
            if os.path.exists(folder):
                docx_files.extend([
                    (os.path.join(folder, f), f) for f in os.listdir(folder) 
                    if f.endswith('.docx')
                ])
        
        if docx_files:
            self.logger.info(f" Found {len(docx_files)} DOCX files")
            
            for docx_path, docx_file in docx_files:
                self.stats['total_files'] += 1
                
                text, metadata = self.process_single_document(docx_path)
                
                if text and 'error' not in metadata:
                    document_key = f"DOCX_{docx_file}"
                    all_documents[document_key] = text
                    all_metadata[document_key] = metadata
                    self.stats['processed_files'] += 1
                    self.stats['total_characters'] += len(text)
                    self.logger.info(f" Processed: {docx_file}")
                else:
                    self.stats['failed_files'] += 1
                    error_msg = metadata.get('error', 'Unknown error')
                    self.stats['processing_errors'].append(f"{docx_file}: {error_msg}")
                    self.logger.error(f" Failed: {docx_file}")
        
        # Create combined document
        combined_text = self.create_combined_document(all_documents)
        
        # Save processed documents
        self.save_processed_documents(all_documents, all_metadata, combined_text)
        
        # Log final statistics
        self.log_processing_statistics()
        
        return {
            'documents': all_documents,
            'metadata': all_metadata,
            'combined_text': combined_text,
            'statistics': self.stats
        }
    
    def create_combined_document(self, all_documents: Dict[str, str]) -> str:
        """Create a single combined document from all processed documents"""
        self.logger.info(" Creating combined document...")
        
        combined_parts = []
        
        # Add header
        combined_parts.append(f"University of Sargodha - Rules and Regulations")
        combined_parts.append(f"Document Collection - Processed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        combined_parts.append("=" * 80)
        combined_parts.append("")
        
        # Add table of contents
        combined_parts.append("TABLE OF CONTENTS")
        combined_parts.append("-" * 40)
        for i, doc_name in enumerate(all_documents.keys(), 1):
            combined_parts.append(f"{i}. {doc_name}")
        combined_parts.append("")
        combined_parts.append("=" * 80)
        combined_parts.append("")
        
        # Add each document
        for doc_name, doc_text in all_documents.items():
            if doc_text.strip():  # Only add non-empty documents
                combined_parts.append(f"\n\n{'=' * 80}")
                combined_parts.append(f"DOCUMENT: {doc_name}")
                combined_parts.append(f"{ '=' * 80}\n")
                combined_parts.append(doc_text)
                combined_parts.append(f"\n{'=' * 80}")
                combined_parts.append(f"END OF DOCUMENT: {doc_name}")
                combined_parts.append(f"{ '=' * 80}\n")
        
        combined_text = '\n'.join(combined_parts)
        
        self.logger.info(f" Combined document created: {len(combined_text)} characters")
        return combined_text
    
    def save_processed_documents(self, all_documents: Dict[str, str], 
                               all_metadata: Dict[str, Dict], 
                               combined_text: str):
        """Save all processed documents and metadata"""
        
        # Save combined document
        combined_path = os.path.join(self.processed_folder, "university_rules_combined.txt")
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        self.logger.info(f" Combined document saved: {combined_path}")
        
        # Save individual documents
        individual_folder = os.path.join(self.processed_folder, "individual_documents")
        os.makedirs(individual_folder, exist_ok=True)
        
        for doc_name, doc_text in all_documents.items():
            if doc_text.strip():
                doc_path = os.path.join(individual_folder, f"{doc_name}.txt")
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(doc_text)
        
        # Save metadata
        metadata_path = os.path.join(self.processed_folder, "processing_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'processing_date': datetime.now().isoformat(),
                'statistics': self.stats,
                'documents_metadata': all_metadata
            }, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f" Metadata saved: {metadata_path}")
    
    def log_processing_statistics(self):
        """Log detailed processing statistics"""
        self.logger.info("\n PROCESSING STATISTICS")
        self.logger.info("=" * 50)
        self.logger.info(f"Total files found: {self.stats['total_files']}")
        self.logger.info(f"Successfully processed: {self.stats['processed_files']}")
        self.logger.info(f"Failed to process: {self.stats['failed_files']}")
        self.logger.info(f"Success rate: {(self.stats['processed_files']/max(1,self.stats['total_files']))*100:.1f}%")
        self.logger.info(f"Total characters extracted: {self.stats['total_characters']:,}")
        self.logger.info(f"Average characters per document: {self.stats['total_characters']//max(1,self.stats['processed_files']):,}")
        
        if self.stats['processing_errors']:
            self.logger.warning(f"\n  PROCESSING ERRORS ({len(self.stats['processing_errors'])}):")
            for error in self.stats['processing_errors']:
                self.logger.warning(f"  - {error}")
    
    def get_processing_summary(self) -> Dict:
        """Get summary of document processing results"""
        return {
            'statistics': self.stats,
            'files_locations': {
                'combined_document': os.path.join(self.processed_folder, "university_rules_combined.txt"),
                'individual_documents': os.path.join(self.processed_folder, "individual_documents"),
                'metadata': os.path.join(self.processed_folder, "processing_metadata.json")
            }
        }
