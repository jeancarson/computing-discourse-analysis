"""
Embedding-based Similarity Analysis Between Eras

Computes topic-level similarity using cached embeddings and CSV outputs.
Designed to work standalone without running the full BERTopic pipeline.
"""

import ast
import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EmbeddingSimilarityAnalyzer:
    """
    Analyze topic similarity between Era1 and Era2 using embeddings.
    
    Workflow:
    1. Load cached embeddings (from main pipeline)
    2. Load metadata (to map chunks to documents)
    3. Load topic assignments from CSV
    4. Compute topic centroids
    5. Use visualizer to create heatmaps
    """
    
    def __init__(self, embeddings_cache_dir: Path, embedding_model_name: str = "Alibaba-NLP/gte-base-en-v1.5"):
        self.embeddings_cache_dir = Path(embeddings_cache_dir)
        self.embedding_model_name = embedding_model_name
        self.embeddings = {}
        self.metadata = {}
        self.topic_centroids = {}
        self.topic_ids = {}
        self.topic_keywords = {}
        
    def _get_cache_path(self, era: int) -> Path:
        """Get the cache file path for a specific era's embeddings."""
        slug = self.embedding_model_name.replace('/', '_').replace('-', '_')
        return self.embeddings_cache_dir / f'era{era}_embeddings_{slug}.npy'
    
    def load_embeddings(self, era: int) -> np.ndarray:
        """Load cached embeddings for an era."""
        cache_path = self._get_cache_path(era)
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {cache_path}\n"
                "Run the main pipeline first to generate embeddings."
            )
        
        embeddings = np.load(cache_path)
        self.embeddings[era] = embeddings
        logger.info(f"Loaded Era{era}: {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]})")
        return embeddings
    
    def load_metadata(self, data_root: Path, era: int) -> pd.DataFrame:
        """Load and chunk metadata to match embeddings."""
        from topic_modelling.utils.data_utils import load_processed_data
        from topic_modelling.utils.chunker import SemanticChunker
        
        texts, metadata_list = load_processed_data(data_root, era)
        
        chunker = SemanticChunker(
            max_tokens=5000,
            overlap_tokens=200,
            chunk_threshold=7000
        )
        _, chunked_metadata = chunker.chunk_documents(texts, metadata_list)
        
        metadata_df = pd.DataFrame(chunked_metadata)
        self.metadata[era] = metadata_df
        logger.info(f"Loaded metadata for {len(metadata_df)} Era{era} chunks")
        return metadata_df
    
    def _load_chunk_topics(self, era: int, output_dir: Path) -> np.ndarray:
        """Map chunks to topics using document-level CSV."""
        doc_topics_path = output_dir / f'era{era}' / f'Era{era}_document_topics.csv'
        doc_topics_df = pd.read_csv(doc_topics_path)
        
        # For multi-topic docs, use highest probability topic
        doc_to_topic = doc_topics_df.loc[
            doc_topics_df.groupby('doc_id')['probability'].idxmax()
        ][['doc_id', 'topic']].set_index('doc_id')['topic'].to_dict()
        
        chunk_topics = []
        for _, row in self.metadata[era].iterrows():
            doc_id = row.get('doc_id', row.get('filepath'))
            chunk_topics.append(doc_to_topic.get(doc_id, -1))
        
        return np.array(chunk_topics)
    
    def _load_topic_keywords(self, era: int, output_dir: Path, top_n: int = 5) -> Dict[int, list]:
        """Load topic keywords from CSV."""
        keywords_path = output_dir / f'era{era}' / f'Era{era}_topic_keywords.csv'
        keywords_df = pd.read_csv(keywords_path)
        
        topic_keywords = {}
        for _, row in keywords_df.iterrows():
            topic_id = row['Topic']
            keywords = row['Keywords']
            if isinstance(keywords, str):
                keywords = ast.literal_eval(keywords)
            topic_keywords[topic_id] = keywords[:top_n]
        
        return topic_keywords
    
    def compute_topic_centroids_from_csv(self, output_dir: Path):
        """
        Compute topic centroids using CSV files.
        
        This enables topic similarity analysis without running the full pipeline.
        Requires embeddings and metadata to be loaded first.
        """
        for era in [1, 2]:
            if era not in self.embeddings:
                raise ValueError(f"Load embeddings for Era{era} first")
            if era not in self.metadata:
                raise ValueError(f"Load metadata for Era{era} first")
            
            chunk_topics = self._load_chunk_topics(era, output_dir)
            self.topic_keywords[era] = self._load_topic_keywords(era, output_dir)
            
            embeddings = self.embeddings[era]
            centroids = {}
            
            for topic_id in np.unique(chunk_topics):
                if topic_id == -1:  # Skip outliers
                    continue
                mask = chunk_topics == topic_id
                centroids[topic_id] = np.mean(embeddings[mask], axis=0)
            
            self.topic_centroids[era] = centroids
            self.topic_ids[era] = sorted(centroids.keys())
            logger.info(f"Computed {len(centroids)} topic centroids for Era{era}")
    
    def get_topic_keywords(self, era: int, topic_id: int, top_n: int = 5) -> list:
        """Get keywords for a topic."""
        if era in self.topic_keywords and topic_id in self.topic_keywords[era]:
            return self.topic_keywords[era][topic_id][:top_n]
        return []
