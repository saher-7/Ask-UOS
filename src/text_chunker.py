import re
import tiktoken
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

@dataclass
class TextChunk:
    """Data class for text chunks with comprehensive metadata"""
    content: str
    chunk_id: int
    start_char: int
    end_char: int
    token_count: int
    word_count: int
    source_document: str = ""
    chunk_type: str = "content"  # content, header, table, list
    section_title: str = ""
    relevance_score: float = 1.0

class UniversityTextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 model_name: str = "gpt-3.5-turbo"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")  # Fallback
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/chunking.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # University-specific patterns for better chunking
        self.section_patterns = [
            r'(?i)^(CHAPTER|SECTION|ARTICLE|RULE|REGULATION)\s+\d+',
            r'(?i)^(ADMISSION|EXAMINATION|FEES?|HOSTEL|LIBRARY|DISCIPLINARY)',
            r'(?i)^(UNDERGRADUATE|POSTGRADUATE|GRADUATE|PhD|MS|MSc|MPhil)',
            r'(?i)^(ACADEMIC|SEMESTER|ANNUAL|COURSE|DEGREE|DIPLOMA)'
        ]
        
        # Chunking statistics
        self.stats = {
            'total_characters': 0,
            'total_chunks': 0,
            'average_chunk_size': 0,
            'average_tokens_per_chunk': 0,
            'chunk_size_distribution': {},
            'processing_time': 0
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer"""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
    
    def count_words(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())
    
    def clean_university_text(self, text: str) -> str:
        """University-specific text cleaning"""
        if not text:
            return ""
        
        self.logger.info(" Cleaning university document text...")
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix common university document issues
        # Remove page markers
        text = re.sub(r'--- PAGE \d+ ---', '', text)
        text = re.sub(r'Page \d+ of \d+', '', text)
        
        # Fix rule numbering formatting
        text = re.sub(r'(\d+)\s*\.\s*(\d+)\s*\.\s*(\d+)', r'\1.\2.\3', text)
        
        # Fix common abbreviations
        abbreviation_fixes = {
            'U n i v e r s i t y': 'University',
            'R u l e s': 'Rules',
            'R e g u l a t i o n s': 'Regulations',
            'A d m i s s i o n': 'Admission',
            'E x a m i n a t i o n': 'Examination'
        }
        
        for wrong, right in abbreviation_fixes.items():
            text = text.replace(wrong, right)
        
        # Clean up bullet points and numbering
        text = re.sub(r'^\s*[·]\s*', ' ', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*(\d+)\s*[\.)]\s*', r'\1. ', text, flags=re.MULTILINE)
        
        # Remove duplicate spaces after punctuation
        text = re.sub(r'([.!?])\s+', r'\1 ', text)
        text = re.sub(r'([,;:])\s+', r'\1 ', text)
        
        # Fix common OCR errors in university documents
        ocr_fixes = {
            'Umversity': 'University',
            'Sargodha': 'Sargodha',  # Ensure correct spelling
            'admissi0n': 'admission',
            'examinati0n': 'examination',
            'regulati0ns': 'regulations'
        }
        
        for wrong, right in ocr_fixes.items():
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', right, text, flags=re.IGNORECASE)
        
        self.logger.info(" Text cleaning completed")
        return text.strip()
    
    def identify_section_boundaries(self, text: str) -> List[Tuple[int, str, str]]:
        """Identify natural section boundaries in university documents"""
        boundaries = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # Check against section patterns
            for pattern in self.section_patterns:
                if re.match(pattern, line_stripped):
                    char_position = sum(len(lines[j]) + 1 for j in range(i))
                    section_type = self._classify_section_type(line_stripped)
                    boundaries.append((char_position, line_stripped, section_type))
                    break
        
        return boundaries
    
    def _classify_section_type(self, line: str) -> str:
        """Classify the type of section based on content"""
        line_lower = line.lower()
        
        if any(word in line_lower for word in ['chapter', 'section', 'article']):
            return 'major_section'
        elif any(word in line_lower for word in ['rule', 'regulation']):
            return 'rule'
        elif any(word in line_lower for word in ['admission', 'examination', 'fees']):
            return 'procedure'
        elif any(word in line_lower for word in ['undergraduate', 'postgraduate', 'phd']):
            return 'program_specific'
        else:
            return 'subsection'
    
    def find_optimal_split_point(self, text: str, preferred_end: int, start: int) -> int:
        """Find the optimal point to split text, prioritizing sentence boundaries"""
        
        # Define search window
        search_start = max(start, preferred_end - 300)
        search_end = min(len(text), preferred_end + 300)
        search_text = text[search_start:search_end]
        
        # Look for different types of boundaries in order of preference
        boundary_patterns = [
            # Strong boundaries (paragraph breaks)
            (r'\n\n+', 2),
            # Section boundaries
            (r'\n(?=\d+\.)', 1),
            (r'\n(?=[A-Z][A-Z\s]+:)', 1),
            # Sentence boundaries
            (r'[.!?]\s+(?=[A-Z])', 0.8),
            # Clause boundaries
            (r'[;:]\s+', 0.6),
            # Word boundaries (last resort)
            (r'\s+', 0.3)
        ]
        
        best_split = preferred_end
        best_score = 0
        
        for pattern, score_weight in boundary_patterns:
            matches = list(re.finditer(pattern, search_text))
            
            for match in matches:
                candidate_pos = search_start + match.end()
                
                # Score based on distance from preferred position
                distance = abs(candidate_pos - preferred_end)
                distance_score = max(0, 1 - (distance / 300))
                
                final_score = score_weight * distance_score
                
                if final_score > best_score:
                    best_score = final_score
                    best_split = candidate_pos
        
        return best_split
    
    def create_chunks(self, text: str, source_document: str = "") -> List[TextChunk]:
        """Create optimized chunks for university documents"""
        start_time = datetime.now()
        
        self.logger.info(" Creating chunks for document: {source_document}")
        self.logger.info(f" Parameters: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        # Clean text first
        text = self.clean_university_text(text)
        self.stats['total_characters'] = len(text)
        
        # Identify section boundaries for better chunking
        section_boundaries = self.identify_section_boundaries(text)
        self.logger.info(f" Found {len(section_boundaries)} section boundaries")
        
        chunks = []
        start = 0
        chunk_id = 0
        
        max_chunks = 10000  # Hard limit to prevent runaway loops
        while start < len(text):
            if chunk_id > max_chunks:
                self.logger.error(f"Aborting: chunk_id exceeded {max_chunks}. start={start}, end={end if 'end' in locals() else 'N/A'}")
                break
            # Debug print for loop progress
            self.logger.debug(f"Loop: start={start}, end will be calculated, chunk_id={chunk_id}")
            # Calculate preferred end position
            preferred_end = start + self.chunk_size
            # Don't exceed text length
            if preferred_end >= len(text):
                end = len(text)
            else:
                # Find optimal split point
                end = self.find_optimal_split_point(text, preferred_end, start)
                # Fallback: ensure end always advances
                if end <= start:
                    end = min(start + max(1, self.chunk_size), len(text))
            # SAFETY CHECK: Prevent infinite loop if end does not advance
            if end <= start:
                self.logger.error(f"Stuck in loop: start={start}, end={end}, chunk_id={chunk_id}")
                break
            
            # Extract chunk content
            chunk_content = text[start:end].strip()
            if chunk_id % 1000 == 0:
                print(f"DEBUG: chunk_id={chunk_id}, start={start}, end={end}, chunk_len={len(chunk_content)}")
            
            # Skip tiny chunks (unless it's the last chunk)
            if len(chunk_content) < 100 and end < len(text):
                start = end - self.chunk_overlap
                continue
            
            # Create chunk with metadata
            token_count = self.count_tokens(chunk_content)
            word_count = self.count_words(chunk_content)
            
            # Determine section context
            current_section = self._find_current_section(start, section_boundaries)
            
            chunk = TextChunk(
                content=chunk_content,
                chunk_id=chunk_id,
                start_char=start,
                end_char=end,
                token_count=token_count,
                word_count=word_count,
                source_document=source_document,
                section_title=current_section['title'],
                chunk_type=current_section['type']
            )
            
            chunks.append(chunk)
            
            # Progress logging
            if chunk_id % 50 == 0:
                progress = (start / len(text)) * 100
                self.logger.info(f"Created chunk {chunk_id}: {progress:.1f}% complete")
            
            # Move to next chunk position
            start = end - self.chunk_overlap
            chunk_id += 1
        
        # Calculate statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        self._calculate_chunking_stats(chunks, processing_time)
        
        self.logger.info(f" Created {len(chunks)} chunks in {processing_time:.2f} seconds")
        return chunks
    
    def _find_current_section(self, position: int, boundaries: List[Tuple[int, str, str]]) -> Dict[str, str]:
        """Find the current section context for a given position"""
        current_section = {'title': '', 'type': 'content'}
        
        for boundary_pos, title, section_type in boundaries:
            if boundary_pos <= position:
                current_section = {'title': title, 'type': section_type}
            else:
                break
        
        return current_section
    
    def _calculate_chunking_stats(self, chunks: List[TextChunk], processing_time: float):
        """Calculate detailed chunking statistics"""
        if not chunks:
            return
        
        self.stats['total_chunks'] = len(chunks)
        self.stats['processing_time'] = processing_time
        
        # Size statistics
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        token_counts = [chunk.token_count for chunk in chunks]
        
        self.stats['average_chunk_size'] = sum(chunk_sizes) // len(chunk_sizes)
        self.stats['average_tokens_per_chunk'] = sum(token_counts) // len(token_counts)
        
        # Size distribution
        size_ranges = [
            (0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, float('inf'))
        ]
        
        distribution = {}
        for min_size, max_size in size_ranges:
            count = sum(1 for size in chunk_sizes if min_size <= size < max_size)
            range_label = f"{min_size}-{max_size if max_size != float('inf') else '2000+'}"
            distribution[range_label] = count
        
        self.stats['chunk_size_distribution'] = distribution
    
    def save_chunks(self, chunks: List[TextChunk], output_dir: str = "data/processed"):
        """Save chunks with comprehensive metadata"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save chunks as JSON
        chunks_data = {
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'chunk_parameters': {
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'model_name': self.model_name
                },
                'statistics': self.stats
            },
            'chunks': [asdict(chunk) for chunk in chunks]
        }
        
        chunks_json_path = os.path.join(output_dir, "text_chunks.json")
        with open(chunks_json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save human-readable chunks
        chunks_txt_path = os.path.join(output_dir, "text_chunks_readable.txt")
        with open(chunks_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"University of Sargodha - Text Chunks\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total chunks: {len(chunks)}\n")
            f.write("=" * 80 + "\n\n")
            
            for chunk in chunks:
                f.write(f"CHUNK {chunk.chunk_id}\n")
                f.write(f"Source: {chunk.source_document}\n")
                f.write(f"Section: {chunk.section_title}\n")
                f.write(f"Type: {chunk.chunk_type}\n")
                f.write(f"Position: {chunk.start_char}-{chunk.end_char}\n")
                f.write(f"Size: {len(chunk.content)} chars, {chunk.token_count} tokens, {chunk.word_count} words\n")
                f.write("-" * 40 + "\n")
                f.write(f"{chunk.content}\n")
                f.write("=" * 80 + "\n\n")
        
        self.logger.info(f" Chunks saved:")
        self.logger.info(f"  JSON: {chunks_json_path}")
        self.logger.info(f"  Readable: {chunks_txt_path}")
    
    def load_chunks(self, chunks_file: str = "data/processed/text_chunks.json") -> List[TextChunk]:
        """Load chunks from saved JSON file"""
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = []
            for chunk_data in data['chunks']:
                chunk = TextChunk(**chunk_data)
                chunks.append(chunk)
            
            self.logger.info(f" Loaded {len(chunks)} chunks from {chunks_file}")
            return chunks
            
        except Exception as e:
            self.logger.error(f" Error loading chunks: {e}")
            return []
    
    def get_chunking_summary(self) -> Dict:
        """Get summary of chunking results"""
        return {
            'statistics': self.stats,
            'parameters': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'model_name': self.model_name
            },
            'files': {
                'chunks_json': "data/processed/text_chunks.json",
                'chunks_readable': "data/processed/text_chunks_readable.txt"
            }
        }
