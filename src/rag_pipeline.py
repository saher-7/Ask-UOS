import openai
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import json
import time
import re
from src.embedding_service import get_embedding_service
from src.vector_store import UniversityVectorStore
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RAGResponse:
    """Data class for RAG pipeline responses"""
    answer: str
    source_chunks: List[str]
    metadata: List[Dict[str, Any]]
    confidence_scores: List[float]
    query: str
    response_time: float
    total_tokens_used: int
    sources_used: int

class UniversityRAGPipeline:
    """Complete RAG pipeline for University of Sargodha Rules & Regulations"""
    
    def __init__(self, 
                 vector_store: Optional[UniversityVectorStore] = None,
                 embedding_service=None,
                 llm_model: str = "gpt-3.5-turbo",
                 use_local_embeddings: bool = False):
        
        # Initialize components
        self.vector_store = vector_store or UniversityVectorStore()
        
        if embedding_service:
            self.embedding_service = embedding_service
        else:
            self.embedding_service, self.service_type = get_embedding_service(use_openai=not use_local_embeddings)
        
        self.llm_model = llm_model
        
        # Initialize OpenAI client for LLM
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and not use_local_embeddings:
            self.openai_client = openai.OpenAI(api_key=openai_key)
            self.use_openai_llm = True
        else:
            self.openai_client = None
            self.use_openai_llm = False
            print("  No OpenAI API key found. LLM responses will be basic.")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/rag_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # RAG parameters
        self.max_context_chunks = 5
        self.max_context_tokens = 3000
        self.min_similarity_score = 0.3
        
        # Response templates for fallback
        self.fallback_templates = {
            'admission': "For admission-related queries, please contact the University of Sargodha admissions office or visit their official website.",
            'fees': "For fee structure and payment information, please contact the accounts office or check the official fee schedule.",
            'examination': "For examination rules and procedures, please refer to the Controller of Examinations office.",
            'general': "I don't have enough specific information to answer your question accurately. Please contact the relevant university department."
        }
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'successful_responses': 0,
            'fallback_responses': 0,
            'average_response_time': 0.0,
            'total_tokens_used': 0,
            'cache_hits': 0
        }
        
        self.logger.info(" University RAG Pipeline initialized")
        self.logger.info(f" Using {getattr(self, 'service_type', 'unknown')} embedding service")
        self.logger.info(f" LLM available: {self.use_openai_llm}")
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = None,
                               use_category_filter: bool = True) -> Dict[str, Any]:
        """Retrieve relevant document chunks for query with smart filtering"""
        if n_results is None:
            n_results = self.max_context_chunks
        
        self.logger.info(f" Retrieving relevant chunks for: '{query[:50]}...'")
        
        try:
            # Create query embedding
            query_embedding = self.embedding_service.create_embedding(query)
            
            if not query_embedding:
                self.logger.error(" Failed to create query embedding")
                return {'documents': [], 'metadatas': [], 'distances': [], 'scores': []}
            
            # Detect query category for smarter search
            query_category = self._detect_query_category(query)
            
            # First try category-specific search if category detected
            if use_category_filter and query_category != 'general':
                self.logger.info(f" Trying category-specific search: {query_category}")
                results = self.vector_store.search_by_category(
                    query_embedding, query_category, n_results * 2  # Get more for filtering
                )
                
                # If we get good results, use them; otherwise fall back to general search
                if results['total_found'] >= 2 and any(score >= self.min_similarity_score for score in results.get('scores', [])):
                    self.logger.info(f" Found {results['total_found']} category-specific results")
                else:
                    self.logger.info(" Falling back to general search")
                    results = self.vector_store.similarity_search(query_embedding, n_results)
            else:
                # General search
                results = self.vector_store.similarity_search(query_embedding, n_results)
            
            # Filter results by similarity score
            filtered_results = self._filter_by_similarity(results)
            
            self.logger.info(f" Retrieved {len(filtered_results['documents'])} relevant chunks")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f" Error retrieving chunks: {e}")
            return {'documents': [], 'metadatas': [], 'distances': [], 'scores': []}
    
    def _detect_query_category(self, query: str) -> str:
        """Detect the category of the query for smart filtering"""
        query_lower = query.lower()
        
        category_keywords = {
            'admission': ['admission', 'apply', 'application', 'eligibility', 'entrance', 'merit', 'selection'],
            'examination': ['exam', 'test', 'assessment', 'evaluation', 'grade', 'marks', 'result', 'paper'],
            'fees': ['fee', 'cost', 'payment', 'tuition', 'charges', 'financial', 'money', 'amount'],
            'academic': ['course', 'curriculum', 'credit', 'semester', 'degree', 'program', 'subject'],
            'discipline': ['discipline', 'misconduct', 'violation', 'penalty', 'suspension', 'punishment'],
            'hostel': ['hostel', 'residence', 'accommodation', 'dormitory', 'room', 'boarding'],
            'library': ['library', 'book', 'borrowing', 'reading', 'reference']
        }
        
        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or 'general' if no strong match
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] >= 1:  # At least one keyword match
                return best_category
        
        return 'general'
    
    def _filter_by_similarity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Filter results by minimum similarity score"""
        if not results.get('scores'):
            return results
        
        # Find indices of results that meet minimum similarity
        good_indices = [
            i for i, score in enumerate(results['scores']) 
            if score >= self.min_similarity_score
        ]
        
        if not good_indices:
            # If no results meet threshold, take the best one
            best_idx = results['scores'].index(max(results['scores']))
            good_indices = [best_idx]
        
        # Filter all result arrays
        filtered = {}
        for key in results:
            if isinstance(results[key], list) and results[key]:
                filtered[key] = [results[key][i] for i in good_indices]
            else:
                filtered[key] = results[key]
        
        return filtered
    
    def prepare_context(self, chunks: List[str], metadatas: List[Dict], 
                       max_tokens: int = None) -> Tuple[str, List[str]]:
        """Prepare context string from retrieved chunks with source tracking"""
        if max_tokens is None:
            max_tokens = self.max_context_tokens
        
        context_parts = []
        sources_used = []
        current_tokens = 0
        
        for i, (chunk, metadata) in enumerate(zip(chunks, metadatas)):
            # Estimate token count (rough approximation: 1 token ≈ 4 characters)
            chunk_tokens = len(chunk) // 4
            
            if current_tokens + chunk_tokens > max_tokens and context_parts:
                break
            
            # Format chunk with source information
            source_info = f"[Source: {metadata.get('section_title', 'University Document')}]"
            formatted_chunk = f"{source_info}\n{chunk}\n"
            
            context_parts.append(formatted_chunk)
            sources_used.append(metadata.get('section_title', f'Chunk {i+1}'))
            current_tokens += chunk_tokens
        
        context = "\n---\n".join(context_parts)
        
        self.logger.info(f" Prepared context with {len(context_parts)} chunks (~{current_tokens} tokens)")
        return context, sources_used
    
    def generate_answer(self, query: str, context: str) -> Tuple[str, int]:
        """Generate answer using LLM with university-specific prompt"""
        
        if not self.use_openai_llm:
            return self._generate_fallback_answer(query, context), 0
        
        # University-specific system prompt
        system_prompt = """You are a helpful AI assistant for the University of Sargodha. You answer questions about university rules, regulations, admission procedures, fee structures, examination policies, and other administrative matters.

Instructions:
1. Answer questions using ONLY the information provided in the context documents
2. Be specific and accurate, citing relevant rules or procedures when possible
3. If the context doesn't contain enough information, clearly state this limitation
4. For admission, fee, or procedural questions, provide step-by-step guidance when possible
5. Always maintain a helpful and professional tone
6. If you're unsure about specific details, recommend contacting the relevant university office
7. Format your response clearly with bullet points or numbered lists when appropriate

Context: The provided information comes from official University of Sargodha documents including rules, regulations, and procedural guidelines."""

        user_prompt = f"""Based on the following official University of Sargodha documents, please answer the question.

Context Documents:
{context}

Question: {query}

Please provide a comprehensive answer based on the information above. If the information is incomplete, please specify what additional details might be needed and suggest which university office to contact."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for factual responses
                max_tokens=800,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens
            
            self.logger.info(f" Generated answer using {tokens_used} tokens")
            return answer, tokens_used
            
        except Exception as e:
            self.logger.error(f" Error generating answer with LLM: {e}")
            return self._generate_fallback_answer(query, context), 0
    
    def _generate_fallback_answer(self, query: str, context: str) -> str:
        """Generate a basic fallback answer when LLM is not available"""
        
        # Detect query category for appropriate template
        category = self._detect_query_category(query)
        
        # Extract first few sentences from context
        context_sentences = context.replace('\n', ' ').split('. ')[:3]
        context_excerpt = '. '.join(context_sentences)
        
        if len(context_excerpt) > 200:
            context_excerpt = context_excerpt[:200] + "..."
        
        # Create basic response
        if context_excerpt.strip():
            answer = f"Based on the available information:\n\n{context_excerpt}\n\n"
            answer += self.fallback_templates.get(category, self.fallback_templates['general'])
        else:
            answer = self.fallback_templates.get(category, self.fallback_templates['general'])
        
        return answer
    
    def process_query(self, query: str, include_sources: bool = True) -> RAGResponse:
        """Main method to process a query through the complete RAG pipeline"""
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        self.logger.info(f" Processing query: '{query}'")
        
        try:
            # Step 1: Retrieve relevant chunks
            retrieval_results = self.retrieve_relevant_chunks(query)
            
            if not retrieval_results['documents']:
                # No relevant documents found
                answer = "I couldn't find relevant information in the University of Sargodha documents to answer your question. Please contact the appropriate university office for specific information."
                
                self.stats['fallback_responses'] += 1
                response_time = time.time() - start_time
                
                return RAGResponse(
                    answer=answer,
                    source_chunks=[],
                    metadata=[],
                    confidence_scores=[],
                    query=query,
                    response_time=response_time,
                    total_tokens_used=0,
                    sources_used=0
                )
            
            # Step 2: Prepare context
            context, sources_used = self.prepare_context(
                retrieval_results['documents'],
                retrieval_results['metadatas']
            )
            
            # Step 3: Generate answer
            answer, tokens_used = self.generate_answer(query, context)
            
            # Step 4: Create response
            response_time = time.time() - start_time
            
            # Update statistics
            self.stats['successful_responses'] += 1
            self.stats['total_tokens_used'] += tokens_used
            self.stats['average_response_time'] = (
                (self.stats['average_response_time'] * (self.stats['total_queries'] - 1) + response_time) / 
                self.stats['total_queries']
            )
            
            response = RAGResponse(
                answer=answer,
                source_chunks=retrieval_results['documents'] if include_sources else [],
                metadata=retrieval_results['metadatas'],
                confidence_scores=retrieval_results.get('scores', []),
                query=query,
                response_time=response_time,
                total_tokens_used=tokens_used,
                sources_used=len(sources_used)
            )
            
            self.logger.info(f" Query processed in {response_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f" Error processing query: {e}")
            
            response_time = time.time() - start_time
            self.stats['fallback_responses'] += 1
            
            return RAGResponse(
                answer=f"I encountered an error while processing your question: {str(e)}. Please try again or contact university support.",
                source_chunks=[],
                metadata=[],
                confidence_scores=[],
                query=query,
                response_time=response_time,
                total_tokens_used=0,
                sources_used=0
            )
    
    def get_conversation_context(self, query: str, conversation_history: List[Dict[str, str]]) -> str:
        """Generate context-aware query considering conversation history"""
        
        if not conversation_history:
            return query
        
        # Get last 3 exchanges for context
        recent_history = conversation_history[-3:]
        
        # Build context-aware query
        context_parts = []
        for exchange in recent_history:
            if 'user' in exchange:
                context_parts.append(f"Previous question: {exchange['user']}")
            if 'assistant' in exchange:
                context_parts.append(f"Previous answer: {exchange['assistant'][:100]}...")
        
        context_str = "\n".join(context_parts)
        
        enhanced_query = f"""Previous conversation context:

    def query(self, query: str, include_sources: bool = True) -> RAGResponse:
        """Alias for process_query to maintain compatibility with interfaces"""
        return self.process_query(query, include_sources)

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics - alias for get_statistics"""
        stats = self.get_statistics()

        # Flatten structure for easier access
        return {
            'queries': {
                'total_queries': stats['pipeline_stats']['total_queries'],
                'successful_responses': stats['pipeline_stats']['successful_responses'],
                'success_rate': float(str(stats['pipeline_stats']['success_rate']).rstrip('%'))
            },
            'performance': {
                'average_response_time': float(str(stats['pipeline_stats']['average_response_time']).rstrip('s')),
                'total_tokens_used': stats['pipeline_stats']['total_tokens_used'],
                'average_tokens_per_query': stats['pipeline_stats']['total_tokens_used'] // max(1, stats['pipeline_stats']['total_queries'])
            },
            'configuration': stats['configuration'],
            'vector_store': stats.get('vector_store_stats', {})
        }
    
{context_str}

Current question: {query}"""
        
        return enhanced_query
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        
        # Get embedding service stats
        embedding_stats = self.embedding_service.get_stats()
        
        # Get vector store stats
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            'pipeline_stats': {
                'total_queries': self.stats['total_queries'],
                'successful_responses': self.stats['successful_responses'],
                'fallback_responses': self.stats['fallback_responses'],
                'success_rate': f"{(self.stats['successful_responses'] / max(1, self.stats['total_queries'])) * 100:.1f}%",
                'average_response_time': f"{self.stats['average_response_time']:.3f}s",
                'total_tokens_used': self.stats['total_tokens_used']
            },
            'embedding_stats': embedding_stats,
            'vector_store_stats': vector_stats,
            'configuration': {
                'embedding_service': getattr(self, 'service_type', 'unknown'),
                'llm_model': self.llm_model,
                'llm_available': self.use_openai_llm,
                'max_context_chunks': self.max_context_chunks,
                'max_context_tokens': self.max_context_tokens,
                'min_similarity_score': self.min_similarity_score
            }
        }