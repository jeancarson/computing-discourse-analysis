import re
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SemanticChunker:
    
    def __init__(self, max_tokens: int = 5000, overlap_tokens: int = 200, chunk_threshold: int = 7000):
        """
        Args:
            max_tokens: Target size for each chunk (default 5000 for GTE-base-en-v1.5)
            overlap_tokens: Overlap between chunks for context continuity
            chunk_threshold: Only chunk documents exceeding this size (default 7000)
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.chunk_threshold = chunk_threshold
    
    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)
    
    def chunk_documents(
        self, 
        texts: List[str], 
        metadata_list: List[Dict]
    ) -> Tuple[List[str], List[Dict]]:
        """
        Chunk large documents while preserving document identity for later aggregation.
        
        Returns chunks with metadata including:
        - doc_id: Original document identifier
        - chunk_index: Position in document
        - total_chunks: Total chunks for this document
        - is_chunked: Whether document was split
        """
        chunked_texts = []
        chunked_metadata = []
        
        for idx, (text, metadata) in enumerate(zip(texts, metadata_list)):
            doc_id = metadata.get('filepath', f'doc_{idx}')
            token_count = self.estimate_tokens(text)
            
            if token_count <= self.chunk_threshold:
                chunked_texts.append(text)
                chunk_meta = metadata.copy()
                chunk_meta.update({
                    'doc_id': doc_id,
                    'chunk_id': f'{doc_id}_chunk_0',
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'is_chunked': False,
                    'token_count': token_count
                })
                chunked_metadata.append(chunk_meta)
            else:
                chunks = self._chunk_by_sentences(text)
                logger.info(f"Chunked {doc_id} ({token_count} tokens) into {len(chunks)} chunks")
                
                for chunk_idx, chunk_text in enumerate(chunks):
                    chunk_meta = metadata.copy()
                    chunk_meta.update({
                        'doc_id': doc_id,
                        'chunk_id': f'{doc_id}_chunk_{chunk_idx}',
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'is_chunked': True,
                        'token_count': self.estimate_tokens(chunk_text)
                    })
                    chunked_texts.append(chunk_text)
                    chunked_metadata.append(chunk_meta)
        
        logger.info(f"Chunking: {len(texts)} docs -> {len(chunked_texts)} chunks")
        return chunked_texts, chunked_metadata
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Split text into chunks by sentence boundaries with overlap."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sent in sentences:
            sent_tokens = self.estimate_tokens(sent)
            
            if sent_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                words = sent.split()
                for i in range(0, len(words), self.max_tokens):
                    chunk_words = words[i:i + self.max_tokens]
                    chunks.append(' '.join(chunk_words))
                continue
            
            if current_tokens + sent_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    
                    overlap_sents = []
                    overlap_tokens = 0
                    for s in reversed(current_chunk):
                        s_tokens = self.estimate_tokens(s)
                        if overlap_tokens + s_tokens <= self.overlap_tokens:
                            overlap_sents.insert(0, s)
                            overlap_tokens += s_tokens
                        else:
                            break
                    
                    current_chunk = overlap_sents
                    current_tokens = overlap_tokens
            
            current_chunk.append(sent)
            current_tokens += sent_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    @staticmethod
    def aggregate_chunk_topics(
        chunk_topics: List[int],
        chunk_probs: List[float],
        chunk_metadata: List[Dict],
        strategy: str = 'majority'
    ) -> Dict[str, Dict]:
        """
        Aggregate chunk-level topics back to document-level.
        
        Args:
            chunk_topics: Topic assignments for each chunk
            chunk_probs: Probabilities for each chunk
            chunk_metadata: Metadata including doc_id for each chunk
            strategy: 'majority', 'weighted', or 'multi'
        
        Returns:
            Dict mapping doc_id to {topic, probability, chunk_count}
        """
        from collections import defaultdict
        
        doc_topics = defaultdict(list)
        
        for topic, prob, meta in zip(chunk_topics, chunk_probs, chunk_metadata):
            doc_id = meta.get('doc_id')
            doc_topics[doc_id].append((topic, prob))
        
        results = {}
        for doc_id, topic_probs in doc_topics.items():
            if strategy == 'majority':
                topics = [t for t, _ in topic_probs]
                final_topic = max(set(topics), key=topics.count)
                final_prob = sum(p for t, p in topic_probs if t == final_topic) / len(topic_probs)
            elif strategy == 'weighted':
                topic_weights = defaultdict(float)
                for topic, prob in topic_probs:
                    topic_weights[topic] += prob
                final_topic = max(topic_weights, key=topic_weights.get)
                final_prob = topic_weights[final_topic] / len(topic_probs)
            else:
                final_topic = [t for t, _ in topic_probs]
                final_prob = [p for _, p in topic_probs]
            
            results[doc_id] = {
                'topic': final_topic,
                'probability': final_prob,
                'chunk_count': len(topic_probs)
            }
        
        return results
