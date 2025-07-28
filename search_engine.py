import numpy as np
import faiss
import gc
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self, model_paths: Dict):
        self.model_paths = model_paths
        self.bi_encoder, self.cross_encoder = None, None
        self.chunks, self.chunk_embeddings, self.index = [], None, None

    def _load_bi_encoder(self):
        if self.bi_encoder is None: 
            logger.info("Loading bi-encoder model...")
            self.bi_encoder = SentenceTransformer(self.model_paths['bi_encoder'], device='cpu')

    def _load_cross_encoder(self):
        if self.cross_encoder is None: 
            logger.info("Loading cross-encoder model...")
            self.cross_encoder = CrossEncoder(self.model_paths['cross_encoder'], device='cpu')
            
    def release_models(self): 
        logger.info("Releasing models from memory...")
        self.bi_encoder, self.cross_encoder = None, None
        gc.collect()

    def build_index(self, chunks: List[Dict]):
        if not chunks: 
            raise ValueError("Cannot build index from empty chunks.")
        
        logger.info(f"Building search index for {len(chunks)} chunks...")
        self.chunks = chunks
        self._load_bi_encoder()
        
        chunk_texts = [c['text'] for c in self.chunks]
        self.chunk_embeddings = self.bi_encoder.encode(
            chunk_texts, 
            batch_size=32, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        self.index = faiss.IndexFlatL2(self.chunk_embeddings.shape[1])
        self.index.add(self.chunk_embeddings.astype('float32'))
        
        logger.info(f"Search index built successfully with {self.index.ntotal} vectors")

    def search_and_rerank(self, query: str, top_k_retrieval: int, top_k_rerank: int) -> List[Dict]:
        if self.index is None: 
            raise RuntimeError("Index not built. Call build_index() first.")
        
        logger.info(f"Searching for: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        self._load_bi_encoder()
        self._load_cross_encoder()
        
        query_embedding = self.bi_encoder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k_retrieval)
        
        candidate_items = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                candidate_items.append({
                    'chunk': self.chunks[idx], 
                    'original_index': idx,
                    'initial_distance': float(distances[0][i])
                })
        
        if not candidate_items:
            logger.warning("No candidate items found during retrieval")
            return []
        
        logger.info(f"Re-ranking {len(candidate_items)} candidates...")
        pairs = [[query, item['chunk']['text']] for item in candidate_items]
        
        try:
            rerank_scores = self.cross_encoder.predict(pairs)
            # Normalize rerank scores to [0, 1]
            rerank_scores = (rerank_scores - rerank_scores.min()) / (rerank_scores.max() - rerank_scores.min() + 1e-8)
            
            for i, item in enumerate(candidate_items): 
                item['rerank_score'] = float(rerank_scores[i])
            
            reranked_items = sorted(candidate_items, key=lambda x: x['rerank_score'], reverse=True)
            final_items = reranked_items[:top_k_rerank]
            
            logger.info(f"Reranking complete. Top score: {final_items[0]['rerank_score']:.3f}")
            return final_items
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return sorted(candidate_items, key=lambda x: x['initial_distance'])[:top_k_rerank]