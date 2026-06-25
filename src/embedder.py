import logging
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers is not available. TF-IDF fallback will be used.")

class LRUCache:
    def __init__(self, capacity=1024):
        self.capacity = capacity
        self.cache = {}
        self.keys = []
        
    def get(self, key):
        if key in self.cache:
            self.keys.remove(key)
            self.keys.append(key)
            return self.cache[key]
        return None
        
    def put(self, key, value):
        if key in self.cache:
            self.keys.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest_key = self.keys.pop(0)
            del self.cache[oldest_key]
        self.cache[key] = value
        self.keys.append(key)

class DocumentEmbedder:
    def __init__(self, use_local_transformer=True):
        self.use_local_transformer = use_local_transformer and SENTENCE_TRANSFORMERS_AVAILABLE
        self.model = None
        self.vectorizer = None
        self.embedding_mode = None
        
        # LRU Local Cache Control to manage host memory consumption
        self._cache = LRUCache(capacity=1024)
        
        if self.use_local_transformer:
            try:
                logger.info("Initializing SentenceTransformer ('all-MiniLM-L6-v2')...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_mode = "sentence-transformers"
                logger.info("SentenceTransformer initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {str(e)}. Falling back to TF-IDF.")
                self.model = None
                self.use_local_transformer = False
                
        if not self.use_local_transformer:
            logger.info("Initializing Scikit-Learn TF-IDF Vectorizer...")
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.embedding_mode = "tfidf"
            
    def _get_text_hash(self, text):
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        
    def compute_embeddings(self, corpus):
        """
        Computes embeddings for a list of texts (corpus).
        Uses LRU caching to avoid redundant calculation and manage host memory.
        Returns:
            list of vectors (numpy arrays), and the mode string used.
        """
        uncached_texts = []
        uncached_indices = []
        embeddings = [None] * len(corpus)
        
        # 1. Check cache first
        for idx, text in enumerate(corpus):
            text_hash = self._get_text_hash(text)
            cache_key = (text_hash, self.embedding_mode)
            cached_val = self._cache.get(cache_key)
            if cached_val is not None:
                embeddings[idx] = cached_val
            else:
                uncached_texts.append(text)
                uncached_indices.append(idx)
                
        # 2. Compute embeddings for uncached texts
        if uncached_texts:
            computed_embeddings = []
            
            if self.embedding_mode == "sentence-transformers" and self.model is not None:
                try:
                    logger.info(f"Generating embeddings for {len(uncached_texts)} uncached documents using SentenceTransformer...")
                    computed_embeddings = self.model.encode(uncached_texts, show_progress_bar=False)
                except Exception as e:
                    logger.error(f"SentenceTransformer encoding failed: {str(e)}. Switching to TF-IDF fallback...")
                    self.vectorizer = TfidfVectorizer(stop_words='english')
                    self.embedding_mode = "tfidf"
                    uncached_texts = corpus
                    uncached_indices = list(range(len(corpus)))
                    embeddings = [None] * len(corpus)
            
            if self.embedding_mode == "tfidf":
                logger.info(f"Generating TF-IDF embeddings for {len(uncached_texts)} documents...")
                try:
                    tfidf_matrix = self.vectorizer.fit_transform(corpus)
                    computed_embeddings = tfidf_matrix.toarray()
                    for i, emb in enumerate(computed_embeddings):
                        text_hash = self._get_text_hash(corpus[i])
                        self._cache.put((text_hash, "tfidf"), emb)
                        embeddings[i] = emb
                    return embeddings, self.embedding_mode
                except Exception as e:
                    logger.error(f"TF-IDF encoding failed: {str(e)}")
                    raise RuntimeError(f"All embedding approaches failed. Error: {str(e)}")
            
            # Save newly computed embeddings to cache
            for idx_in_uncached, original_idx in enumerate(uncached_indices):
                emb = computed_embeddings[idx_in_uncached]
                text_hash = self._get_text_hash(uncached_texts[idx_in_uncached])
                self._cache.put((text_hash, "sentence-transformers"), emb)
                embeddings[original_idx] = emb
                
        return embeddings, self.embedding_mode
