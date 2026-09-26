"""
Topic modelling utilities using BERTopic
"""

import logging
from pathlib import Path
from typing import List, Tuple
import traceback

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

from config import TopicmodellingConfig

logger = logging.getLogger(__name__)


class TopicModeler:
    """Handle BERTopic modelling operations"""

    def __init__(self, config: TopicmodellingConfig):
        self.config = config
        self.model = None
        self.documents = None
        self.metadata = None

    def create_model(self) -> BERTopic:
        """Create BERTopic model with configured components"""
        # Initialize components
        embedding_model = SentenceTransformer(self.config.embedding_model, trust_remote_code=True)

        umap_model = UMAP(
            n_neighbors=self.config.umap.n_neighbors,
            n_components=self.config.umap.n_components,
            min_dist=self.config.umap.min_dist,
            random_state=self.config.umap.random_state
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=self.config.hdbscan.min_cluster_size,
            min_samples=self.config.hdbscan.min_samples,
            gen_min_span_tree=self.config.hdbscan.gen_min_span_tree,
            prediction_data=self.config.hdbscan.prediction_data
        )

        # We perform stopword removal in preprocessing; do not remove here.
        vectorizer_model = CountVectorizer(
            ngram_range=self.config.vectorizer.ngram_range,
            stop_words=None
        )

        # Create BERTopic model
        self.model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            top_n_words=self.config.bertopic.top_n_words,
            language=self.config.bertopic.language,
            calculate_probabilities=self.config.bertopic.calculate_probabilities,
            verbose=self.config.bertopic.verbose
        )

        return self.model

    def fit_transform(self, texts: List[str], embeddings=None, metadata: List[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fit BERTopic model and transform texts
        
        Args:
            texts: List of documents to fit and transform
            embeddings: Optional pre-computed embeddings (avoids re-embedding)
            metadata: Optional metadata for documents (used for representative doc names)
        """
        if self.model is None:
            self.create_model()

        # Store documents and metadata for later use
        self.documents = texts
        self.metadata = metadata

        logger.info("Fitting BERTopic model...")
        topics, probs = self.model.fit_transform(texts, embeddings=embeddings)

        return topics, probs

    def get_topic_info(self) -> pd.DataFrame:
        """Get topic information with document names instead of full text"""
        topic_info = self.model.get_topic_info()
        
        # Replace Representative_Docs column with document names if metadata is available
        if self.metadata is not None and 'Representative_Docs' in topic_info.columns:
            # Create a mapping from document text to document name
            doc_to_name = {}
            for i, (doc, meta) in enumerate(zip(self.documents, self.metadata)):
                # Use doc_id or filepath as the name
                name = meta.get('doc_id', meta.get('filepath', f'doc_{i}'))
                # Store mapping for exact match
                doc_to_name[doc] = name
            
            # Replace full text with names in Representative_Docs
            def replace_docs(docs_list):
                if isinstance(docs_list, list):
                    names = []
                    for doc in docs_list:
                        # Try exact match first, fallback to truncated text
                        name = doc_to_name.get(doc, doc[:100] + '...')
                        names.append(name)
                    return names
                return docs_list
            
            topic_info['Representative_Docs'] = topic_info['Representative_Docs'].apply(replace_docs)
        
        return topic_info

    def get_topic_keywords(self) -> pd.DataFrame:
        """Get topic keywords"""
        topic_info = self.get_topic_info()

        # Extract keywords for each topic
        keywords_data = []
        for topic_id in topic_info['Topic']:
            if topic_id != -1:  # Skip outlier topic
                keywords = self.model.get_topic(topic_id)
                if keywords:
                    keywords_data.append({
                        'Topic': topic_id,
                        'Keywords': [word for word, _ in keywords],
                        'Weights': [weight for _, weight in keywords]
                    })

        return pd.DataFrame(keywords_data)

    def save_visualizations(self, output_dir: Path, era: str):
        """Generate and save visualizations"""
        logger.info(f"Attempting to generate visualizations for {era}")

        # Check if we have enough topics for meaningful visualizations
        topic_info = self.get_topic_info()
        num_topics = len(topic_info[topic_info['Topic'] != -1])  # Exclude outliers
        logger.info(f"{era} has {num_topics} topics (excluding outliers)")

        if num_topics < 5:
            logger.warning(f"Skipping visualizations for {era}: Only {num_topics} topics found. "
                          "Visualizations require at least 5 topics for stable plotting.")
            return

        try:
            # Topic visualization
            logger.info(f"Generating topic visualization for {era}")
            fig_topics = self.model.visualize_topics()
            if fig_topics is not None:
                fig_topics.write_html(output_dir / f"{era}_topic_visualization.html")
                logger.info(f"Saved topic visualization for {era}")
            else:
                logger.warning(f"Topic visualization returned None for {era}")

            # Topic hierarchy
            logger.info(f"Generating topic hierarchy for {era}")
            fig_hierarchy = self.model.visualize_hierarchy()
            if fig_hierarchy is not None:
                fig_hierarchy.write_html(output_dir / f"{era}_topic_hierarchy.html")
                logger.info(f"Saved topic hierarchy for {era}")
            else:
                logger.warning(f"Topic hierarchy returned None for {era}")

            # Topic barchart
            logger.info(f"Generating topic barchart for {era}")
            fig_barchart = self.model.visualize_barchart()
            if fig_barchart is not None:
                fig_barchart.write_html(output_dir / f"{era}_topic_barchart.html")
                logger.info(f"Saved topic barchart for {era}")
            else:
                logger.warning(f"Topic barchart returned None for {era}")

            logger.info(f"Completed visualizations for {era}")

        except Exception as e:
            logger.error(f"Error generating visualizations for {era}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")