import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time

class UniversityVectorStore:
    """Enhanced vector store using Faiss instead of ChromaDB - Windows Compatible"""
    
    def __init__(self, collection_name: str = "su_rules_regulations", 
                 persist_directory: str = "models/faiss_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Faiss components
        self.index = None
        self.documents = []
        self.metadatas = []
        self.embedding_dim = None
        
        # Try to load existing data
        self._load_existing()
        
        # Statistics
        self.stats = {
            'total_documents': len(self.documents),
            'successful_additions': 0,
            'failed_additions': 0,
            'last_updated': None,
            'search_queries': 0,
            'average_search_time': 0.0
        }
        
        print("[OK] Vector store initialized with {} documents".format(len(self.documents)))
    
    def _load_existing(self):
        """Load existing index and metadata"""
        index_path = os.path.join(self.persist_directory, "index.faiss")
        metadata_path = os.path.join(self.persist_directory, "metadata.json")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                self.index = faiss.read_index(index_path)
                self.embedding_dim = self.index.d
                
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', [])
                    self.metadatas = data.get('metadatas', [])
                
                print("[OK] Loaded existing collection: {} ({} documents)".format(
                    self.collection_name, len(self.documents)))
            except Exception as e:
                print("[WARNING] Could not load existing index: {}".format(e))
                self._initialize_empty()
        else:
            self._initialize_empty()
    
    def _initialize_empty(self):
        """Initialize empty collections"""
        self.index = None
        self.documents = []
        self.metadatas = []
        self.embedding_dim = None
        print("[OK] Created new collection: {}".format(self.collection_name))
    
    def add_chunks(self, chunks, embeddings, batch_size: int = 100) -> Dict[str, Any]:
        """Add text chunks with embeddings to vector store in batches"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks ({}) must match number of embeddings ({})".format(
                len(chunks), len(embeddings)))
        
        print("[PROCESSING] Adding {} chunks to vector store...".format(len(chunks)))
        
        results = {
            'successful_additions': 0,
            'failed_additions': 0,
            'errors': []
        }
        
        try:
            # Initialize index if not exists
            if self.index is None:
                self.embedding_dim = len(embeddings[0])
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product similarity
                print("[OK] Created new Faiss index with dimension {}".format(self.embedding_dim))
            
            # Convert embeddings to numpy array
            embeddings_np = np.array(embeddings, dtype=np.float32)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_np)
            
            # Add to index
            self.index.add(embeddings_np)
            
            # Store documents and metadata
            for i, chunk in enumerate(chunks):
                self.documents.append(chunk.content)
                
                metadata = {
                    'chunk_id': chunk.chunk_id,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'token_count': chunk.token_count,
                    'word_count': chunk.word_count,
                    'char_count': len(chunk.content),
                    'source_document': chunk.source_document,
                    'chunk_type': chunk.chunk_type,
                    'section_title': chunk.section_title,
                    'relevance_score': chunk.relevance_score,
                    'added_date': datetime.now().isoformat(),
                    
                    # University-specific metadata
                    'university': 'University of Sargodha',
                    'document_category': self._categorize_content(chunk.content),
                    'contains_rules': self._contains_rules(chunk.content),
                    'contains_fees': self._contains_fees(chunk.content),
                    'contains_admission': self._contains_admission(chunk.content),
                    'contains_examination': self._contains_examination(chunk.content)
                }
                
                self.metadatas.append(metadata)
                
                # Progress update
                if (i + 1) % batch_size == 0:
                    progress = ((i + 1) / len(chunks)) * 100
                    print("\r[PROCESSING] Progress: {:.1f}% ({}/{})".format(progress, i + 1, len(chunks)), end="")
            
            print()  # New line after progress
            
            # Save to disk
            self._save_index()
            
            results['successful_additions'] = len(chunks)
            
        except Exception as e:
            error_msg = "Failed to add chunks: {}".format(e)
            print("[ERROR] {}".format(error_msg))
            results['failed_additions'] = len(chunks)
            results['errors'].append(error_msg)
        
        # Update statistics
        self.stats['successful_additions'] += results['successful_additions']
        self.stats['failed_additions'] += results['failed_additions']
        self.stats['total_documents'] = len(self.documents)
        self.stats['last_updated'] = datetime.now().isoformat()
        
        print("[OK] Successfully added {} chunks".format(results['successful_additions']))
        print("[STATS] Total documents in collection: {}".format(len(self.documents)))
        
        return results
    
    def similarity_search(self, query_embedding: List[float], n_results: int = 5,
                         filters: Optional[Dict[str, Any]] = None,
                         include_scores: bool = True) -> Dict[str, Any]:
        """Enhanced similarity search with filtering and scoring"""
        
        start_time = time.time()
        self.stats['search_queries'] += 1
        
        try:
            if self.index is None or len(self.documents) == 0:
                return {
                    'documents': [], 'metadatas': [], 'distances': [], 
                    'scores': [], 'total_found': 0
                }
            
            # Convert query to numpy array and normalize
            query_np = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_np)
            
            # Search
            search_k = min(n_results * 2, len(self.documents))  # Get more results for filtering
            scores, indices = self.index.search(query_np, search_k)
            
            # Filter results based on filters if provided
            filtered_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= len(self.documents):
                    continue
                
                metadata = self.metadatas[idx]
                
                # Apply filters
                if filters:
                    match = True
                    for key, value in filters.items():
                        if metadata.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue
                
                filtered_results.append({
                    'document': self.documents[idx],
                    'metadata': metadata,
                    'score': float(score),
                    'distance': 1 - float(score)  # Convert similarity to distance
                })
            
            # Limit to requested number of results
            filtered_results = filtered_results[:n_results]
            
            # Prepare response
            processed_results = {
                'documents': [r['document'] for r in filtered_results],
                'metadatas': [r['metadata'] for r in filtered_results],
                'distances': [r['distance'] for r in filtered_results],
                'scores': [r['score'] for r in filtered_results],
                'total_found': len(filtered_results)
            }
            
            # Update search statistics
            search_time = time.time() - start_time
            self.stats['average_search_time'] = (
                (self.stats['average_search_time'] * (self.stats['search_queries'] - 1) + search_time) / 
                self.stats['search_queries']
            )
            
            return processed_results
            
        except Exception as e:
            print("[ERROR] Error during similarity search: {}".format(e))
            return {
                'documents': [], 'metadatas': [], 'distances': [], 
                'scores': [], 'total_found': 0, 'error': str(e)
            }
    
    def search_by_category(self, query_embedding: List[float], category: str, 
                          n_results: int = 5) -> Dict[str, Any]:
        """Search within a specific document category"""
        filters = {'document_category': category}
        return self.similarity_search(query_embedding, n_results, filters)
    
    def search_rules_only(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """Search only in rule-related content"""
        filters = {'contains_rules': True}
        return self.similarity_search(query_embedding, n_results, filters)
    
    def _categorize_content(self, content: str) -> str:
        """Categorize chunk content based on keywords"""
        content_lower = content.lower()
        
        categories = {
            'admission': ['admission', 'apply', 'application', 'eligibility', 'entrance'],
            'examination': ['exam', 'test', 'assessment', 'evaluation', 'grade', 'marks'],
            'fees': ['fee', 'cost', 'payment', 'tuition', 'charges', 'financial'],
            'academic': ['course', 'curriculum', 'credit', 'semester', 'degree', 'program'],
            'discipline': ['discipline', 'misconduct', 'violation', 'penalty', 'suspension'],
            'hostel': ['hostel', 'residence', 'accommodation', 'dormitory'],
            'library': ['library', 'book', 'borrowing', 'reading'],
            'general': []
        }
        
        for category, keywords in categories.items():
            if any(keyword in content_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _contains_rules(self, content: str) -> bool:
        """Check if content contains rule-related information"""
        rule_keywords = ['rule', 'regulation', 'policy', 'guideline', 'procedure', 'requirement']
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in rule_keywords)
    
    def _contains_fees(self, content: str) -> bool:
        """Check if content contains fee-related information"""
        fee_keywords = ['fee', 'cost', 'payment', 'tuition', 'charges', 'amount', 'rupees', 'rs.']
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in fee_keywords)
    
    def _contains_admission(self, content: str) -> bool:
        """Check if content contains admission-related information"""
        admission_keywords = ['admission', 'apply', 'application', 'eligibility', 'merit', 'selection']
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in admission_keywords)
    
    def _contains_examination(self, content: str) -> bool:
        """Check if content contains examination-related information"""
        exam_keywords = ['exam', 'test', 'assessment', 'evaluation', 'grade', 'marks', 'result']
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in exam_keywords)
    
    def _save_index(self):
        """Save index and metadata to disk"""
        try:
            index_path = os.path.join(self.persist_directory, "index.faiss")
            metadata_path = os.path.join(self.persist_directory, "metadata.json")
            
            # Save Faiss index
            if self.index is not None:
                faiss.write_index(self.index, index_path)
            
            # Save metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'collection_name': self.collection_name,
                    'created_date': datetime.now().isoformat(),
                    'total_documents': len(self.documents),
                    'documents': self.documents,
                    'metadatas': self.metadatas,
                    'embedding_dimension': self.embedding_dim
                }, f, ensure_ascii=False, indent=2)
            
            print("[OK] Index and metadata saved to disk")
        except Exception as e:
            print("[WARNING] Could not save index: {}".format(e))
    
    def clear_collection(self):
        """Clear all documents from collection"""
        try:
            self.index = None
            self.documents = []
            self.metadatas = []
            self.embedding_dim = None
            
            # Reset statistics
            self.stats = {
                'total_documents': 0,
                'successful_additions': 0,
                'failed_additions': 0,
                'last_updated': None,
                'search_queries': 0,
                'average_search_time': 0.0
            }
            
            print("[OK] Collection cleared and recreated")
            
        except Exception as e:
            print("[ERROR] Error clearing collection: {}".format(e))
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics"""
        try:
            if len(self.metadatas) == 0:
                return {'total_documents': 0, 'message': 'Collection is empty'}
            
            # Calculate statistics
            sample_size = min(1000, len(self.metadatas))
            sample_metadatas = self.metadatas[:sample_size]
            
            # Category distribution
            categories = {}
            chunk_types = {}
            source_documents = {}
            
            for meta in sample_metadatas:
                # Categories
                category = meta.get('document_category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
                
                # Chunk types
                chunk_type = meta.get('chunk_type', 'unknown')
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                
                # Source documents
                source = meta.get('source_document', 'unknown')
                source_documents[source] = source_documents.get(source, 0) + 1
            
            # Content analysis
            rules_count = sum(1 for meta in sample_metadatas if meta.get('contains_rules', False))
            fees_count = sum(1 for meta in sample_metadatas if meta.get('contains_fees', False))
            admission_count = sum(1 for meta in sample_metadatas if meta.get('contains_admission', False))
            exam_count = sum(1 for meta in sample_metadatas if meta.get('contains_examination', False))
            
            stats = {
                'total_documents': len(self.documents),
                'sample_size': sample_size,
                'embedding_dimension': self.embedding_dim or 0,
                'distributions': {
                    'categories': categories,
                    'chunk_types': chunk_types,
                    'source_documents': source_documents
                },
                'content_analysis': {
                    'contains_rules': "{} ({:.1f}%)".format(rules_count, (rules_count/sample_size)*100),
                    'contains_fees': "{} ({:.1f}%)".format(fees_count, (fees_count/sample_size)*100),
                    'contains_admission': "{} ({:.1f}%)".format(admission_count, (admission_count/sample_size)*100),
                    'contains_examination': "{} ({:.1f}%)".format(exam_count, (exam_count/sample_size)*100)
                },
                'search_stats': {
                    'total_searches': self.stats['search_queries'],
                    'average_search_time': "{:.3f}s".format(self.stats['average_search_time'])
                }
            }
            
            return stats
            
        except Exception as e:
            print("[ERROR] Error getting collection stats: {}".format(e))
            return {'error': str(e)}
    
    def export_collection(self, export_path: str = "data/processed/vector_store_backup.json"):
        """Export collection data for backup"""
        try:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            export_data = {
                'collection_name': self.collection_name,
                'export_date': datetime.now().isoformat(),
                'total_documents': len(self.documents),
                'statistics': self.stats,
                'data': {
                    'documents': self.documents,
                    'metadatas': self.metadatas
                }
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print("[OK] Collection exported to {}".format(export_path))
            return True
            
        except Exception as e:
            print("[ERROR] Error exporting collection: {}".format(e))
            return False