import openai
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
import os
import json
from datetime import datetime
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import hashlib

load_dotenv()

class EmbeddingService:
    """OpenAI-based embedding service with caching and error handling"""
    
    def __init__(self, model_name: str = None):
        # Support for GROQ and OpenAI API keys
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        
        # Set embedding dimension based on model
        if self.model_name == "text-embedding-ada-002":
            self.embedding_dim = 1536
        else:
            self.embedding_dim = 384  # default for MiniLM etc.

        # Always use OpenAI for embeddings
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        else:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file for embeddings.")
        
        # Setup caching
        self.cache_dir = "models/embedding_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"openai_embeddings_{self.model_name}.json")
        self.cache = self._load_cache()

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/embeddings.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'total_tokens_processed': 0,
            'total_cost_estimate': 0.0
        }

    @staticmethod
    def groq_chat_completion(messages, model=None, groq_api_key=None):
        """Call Groq API for chat completions (LLM) only. Use this in your LLM logic, not for embeddings."""
        import requests
        groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        model = model or os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": messages
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    
    def _load_cache(self) -> Dict[str, List[float]]:
        """Load embedding cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save embedding cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            self.logger.warning(f"Could not save cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text with caching"""
        self.stats['total_requests'] += 1
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        try:
            self._rate_limit()
            response = openai.Embedding.create(
                input=text,
                model=self.model_name
            )
            embedding = response.data[0].embedding
            
            # Update statistics
            self.stats['api_calls'] += 1
            self.stats['total_tokens_processed'] += len(text.split())
            self.stats['total_cost_estimate'] += len(text) * 0.0001 / 1000
            
            # Cache the result
            self.cache[cache_key] = embedding
            
            # Save cache periodically
            if self.stats['api_calls'] % 10 == 0:
                self._save_cache()
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error creating embedding: {e}")
            return []
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 100, 
                              show_progress: bool = True) -> List[List[float]]:
        """Create embeddings for multiple texts in batches with progress tracking"""
        if not texts:
            return []
        
        total_batches = (len(texts) - 1) // batch_size + 1
        self.logger.info(f"Creating embeddings for {len(texts)} texts in {total_batches} batches...")
        
        embeddings = []
        
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx:batch_idx + batch_size]
            batch_embeddings = []
            
            # Check cache for each text in batch
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(batch_texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self.cache:
                    batch_embeddings.append(self.cache[cache_key])
                    self.stats['cache_hits'] += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    batch_embeddings.append(None)  # Placeholder
            
            # Make API call for uncached texts
            if uncached_texts:
                try:
                    self._rate_limit()
                    response = openai.Embedding.create(
                        input=uncached_texts,
                        model=self.model_name
                    )
                    
                    # Fill in the uncached embeddings
                    for i, embedding_data in enumerate(response.data):
                        original_idx = uncached_indices[i]
                        embedding = embedding_data.embedding
                        batch_embeddings[original_idx] = embedding
                        
                        # Cache the result
                        cache_key = self._get_cache_key(uncached_texts[i])
                        self.cache[cache_key] = embedding
                    
                    # Update statistics
                    self.stats['api_calls'] += 1
                    self.stats['total_tokens_processed'] += sum(len(text.split()) for text in uncached_texts)
                    self.stats['total_cost_estimate'] += sum(len(text) for text in uncached_texts) * 0.0001 / 1000
                    
                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx//batch_size + 1}: {e}")
                    # Fill failed embeddings with zeros
                    for i in uncached_indices:
                        if batch_embeddings[i] is None:
                            batch_embeddings[i] = [0.0] * self.embedding_dim
            
            embeddings.extend(batch_embeddings)
            
            # Progress update
            if show_progress:
                batch_num = batch_idx // batch_size + 1
                progress = (batch_num / total_batches) * 100
                print(f"\rProgress: {progress:.1f}% ({batch_num}/{total_batches} batches)", end="")
        
        if show_progress:
            print()  # New line after progress
        
        # Save cache
        self._save_cache()
        
        self.logger.info(f"Created {len(embeddings)} embeddings")
        self.logger.info(f"Cache hits: {self.stats['cache_hits']}, API calls: {self.stats['api_calls']}")
        return embeddings
    
    def test_connection(self) -> bool:
        """Test OpenAI API connection"""
        try:
            test_embedding = self.create_embedding("test connection")
            if test_embedding and len(test_embedding) == self.embedding_dim:
                self.logger.info("OpenAI API connection successful")
                return True
            else:
                self.logger.error("OpenAI API returned invalid embedding")
                return False
        except Exception as e:
            self.logger.error(f"OpenAI API connection error: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get embedding service statistics"""
        cache_hit_rate = (self.stats['cache_hits'] / max(1, self.stats['total_requests'])) * 100
        
        return {
            'total_requests': self.stats['total_requests'],
            'cache_hits': self.stats['cache_hits'],
            'api_calls': self.stats['api_calls'],
            'cache_hit_rate': round(cache_hit_rate, 2),
            'total_tokens_processed': self.stats['total_tokens_processed'],
            'estimated_cost_usd': round(self.stats['total_cost_estimate'], 4),
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim
        }


class LocalEmbeddingService:
    """Local embedding service using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/embeddings.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Loading local embedding model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Model loaded successfully (dimension: {self.embedding_dim})")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
        
        # Setup caching
        self.cache_dir = "models/embedding_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"local_embeddings_{model_name.replace('/', '_')}.json")
        self.cache = self._load_cache()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'model_calls': 0,
            'processing_time': 0.0
        }
    
    def _load_cache(self) -> Dict[str, List[float]]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            self.logger.warning(f"Could not save cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5((text + self.model_name).encode('utf-8')).hexdigest()
    
    def create_embedding(self, text: str) -> List[float]:
        self.stats['total_requests'] += 1
        
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        try:
            start_time = time.time()
            embedding = self.model.encode(text)
            processing_time = time.time() - start_time
            
            embedding_list = embedding.tolist()
            
            self.stats['model_calls'] += 1
            self.stats['processing_time'] += processing_time
            
            self.cache[cache_key] = embedding_list
            
            if self.stats['model_calls'] % 50 == 0:
                self._save_cache()
            
            return embedding_list
            
        except Exception as e:
            self.logger.error(f"Error creating embedding: {e}")
            return []
    
    def create_embeddings_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        if not texts:
            return []
        
        self.logger.info(f"Creating embeddings for {len(texts)} texts (local model)...")
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
                self.stats['cache_hits'] += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                embeddings.append(None)
        
        if uncached_texts:
            try:
                start_time = time.time()
                
                if show_progress:
                    print(f"Processing {len(uncached_texts)} new texts with local model...")
                
                uncached_embeddings = self.model.encode(uncached_texts, show_progress_bar=show_progress)
                processing_time = time.time() - start_time
                
                for i, embedding in enumerate(uncached_embeddings):
                    original_idx = uncached_indices[i]
                    embedding_list = embedding.tolist()
                    embeddings[original_idx] = embedding_list
                    
                    cache_key = self._get_cache_key(uncached_texts[i])
                    self.cache[cache_key] = embedding_list
                
                self.stats['model_calls'] += 1
                self.stats['processing_time'] += processing_time
                
                self.logger.info(f"Processed {len(uncached_texts)} texts in {processing_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error in batch processing: {e}")
                for i in uncached_indices:
                    if embeddings[i] is None:
                        embeddings[i] = [0.0] * self.embedding_dim
        
        self._save_cache()
        
        self.logger.info(f"Created {len(embeddings)} embeddings")
        return embeddings
    
    def test_connection(self) -> bool:
        try:
            test_embedding = self.create_embedding("test connection")
            if test_embedding and len(test_embedding) == self.embedding_dim:
                self.logger.info("Local embedding model working correctly")
                return True
            else:
                self.logger.error("Local model returned invalid embedding")
                return False
        except Exception as e:
            self.logger.error(f"Local embedding model error: {e}")
            return False
    
    def get_stats(self) -> Dict:
        cache_hit_rate = (self.stats['cache_hits'] / max(1, self.stats['total_requests'])) * 100
        avg_processing_time = self.stats['processing_time'] / max(1, self.stats['model_calls'])
        
        return {
            'total_requests': self.stats['total_requests'],
            'cache_hits': self.stats['cache_hits'],
            'model_calls': self.stats['model_calls'],
            'cache_hit_rate': round(cache_hit_rate, 2),
            'total_processing_time': round(self.stats['processing_time'], 2),
            'average_processing_time': round(avg_processing_time, 4),
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim
        }


def get_embedding_service(use_openai: bool = None) -> Tuple[object, str]:
    """Factory function to get appropriate embedding service"""
    
    if use_openai is None:
        openai_key = os.getenv("OPENAI_API_KEY")
        use_openai = bool(openai_key)
    
    if use_openai:
        try:
            service = EmbeddingService()
            if service.test_connection():
                return service, "openai"
            else:
                print("OpenAI service failed, falling back to local embeddings")
                use_openai = False
        except Exception as e:
            print(f"OpenAI service error: {e}, falling back to local embeddings")
            use_openai = False
    
    if not use_openai:
        service = LocalEmbeddingService()
        if service.test_connection():
            return service, "local"
        else:
            raise RuntimeError("Both OpenAI and local embedding services failed")