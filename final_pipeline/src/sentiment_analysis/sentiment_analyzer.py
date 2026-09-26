"""
Document-level sentiment analysis using DistilBERT multilingual sentiments
"""

import logging
from typing import List, Dict, Optional
import numpy as np
from transformers import pipeline
import torch

from .utils import chunk_text_by_sentences, aggregate_chunk_sentiments

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analyzer using DistilBERT multilingual model distilled from
    zero-shot classification on the Multilingual Sentiment dataset.
    
    Model: lxyuan/distilbert-base-multilingual-cased-sentiments-student
    - Distilled from MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (teacher)
    - Trained on tyqiangz/multilingual-sentiments (12 languages)
    - 3-class output: positive, neutral, negative
    - ~134M parameters (DistilBERT-base), significantly lighter than RoBERTa-large
    - Broader training domain than tweet-specific models
    
    Attributes:
        model_name (str): HuggingFace model identifier
        device (int): Device for inference (-1 for CPU, 0+ for GPU)
        max_chunk_words (int): Maximum words per chunk (for long documents)
    """
    
    def __init__(
        self,
        model_name: str = "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        device: Optional[int] = None,
        max_chunk_words: int = 400
    ):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: HuggingFace model name
            device: Device for inference (None=auto, -1=CPU, 0+=GPU)
            max_chunk_words: Max words per chunk for long documents
        """
        self.model_name = model_name
        self.max_chunk_words = max_chunk_words
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device
        
        logger.info(f"Loading sentiment model: {model_name}")
        logger.info(f"Device: {'GPU' if self.device >= 0 else 'CPU'}")
        
        # Load model pipeline
        # top_k=None returns probabilities for ALL classes (positive, neutral, negative)
        # This allows computing expected sentiment = P(positive) - P(negative)
        # rather than losing information by only using the top-1 label
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=self.device,
            truncation=True,
            max_length=512,
            top_k=None
        )
        
        logger.info("Sentiment model loaded successfully")
    
    def analyze_document(self, text: str) -> Dict:
        """
        Analyze sentiment of a single document
        
        Handles long documents by:
        1. Splitting into sentence-based chunks
        2. Analyzing each chunk
        3. Aggregating results (majority vote for label, mean score)
        
        Args:
            text: Document text to analyze
        
        Returns:
            Dictionary with:
                - label: 'positive', 'neutral', or 'negative' (top class)
                - score: Confidence score for top class (0-1)
                - compound_score: Expected sentiment = P(pos) - P(neg), range [-1, +1]
                - prob_positive: P(positive)
                - prob_neutral: P(neutral)
                - prob_negative: P(negative)
                - num_chunks: Number of chunks analyzed
        
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> result = analyzer.analyze_document("This is great!")
            >>> result['label']
            'positive'
        """
        if not text or not text.strip():
            logger.warning("Empty text provided")
            return {
                'label': 'neutral', 'score': 0.5, 'compound_score': 0.0,
                'prob_positive': 0.0, 'prob_neutral': 1.0, 'prob_negative': 0.0,
                'num_chunks': 0
            }
        
        # Check if text needs chunking
        word_count = len(text.split())
        
        if word_count <= self.max_chunk_words:
            # Short document - analyze directly
            # top_k=None means result is a list of all class probabilities
            all_scores = self.pipeline(text[:512])[0]
            return self._parse_full_scores(all_scores, num_chunks=1)
        else:
            # Long document - chunk and aggregate
            logger.debug(f"Document has {word_count} words, chunking...")
            chunks = chunk_text_by_sentences(text, self.max_chunk_words)
            
            # Analyze each chunk — get full probability distribution per chunk
            chunk_sentiments = []
            for chunk in chunks:
                if chunk.strip():  # Skip empty chunks
                    all_scores = self.pipeline(chunk[:512])[0]
                    parsed = self._parse_full_scores(all_scores, num_chunks=1)
                    chunk_sentiments.append(parsed)
            
            # Aggregate results across chunks
            aggregated = aggregate_chunk_sentiments(chunk_sentiments)
            aggregated['num_chunks'] = len(chunks)
            
            return aggregated
    
    @staticmethod
    def _parse_full_scores(all_scores: List[Dict], num_chunks: int = 1) -> Dict:
        """
        Parse the full list of class probabilities from the pipeline.
        
        The pipeline with top_k=None returns a list like:
            [{'label': 'neutral', 'score': 0.75},
             {'label': 'positive', 'score': 0.15},
             {'label': 'negative', 'score': 0.10}]
        
        We extract P(pos), P(neu), P(neg) and compute:
            compound_score = P(positive) - P(negative)
        
        This gives a continuous [-1, +1] score where:
            - Strongly positive docs → near +1
            - Strongly negative docs → near -1  
            - Neutral docs → near 0, but NOT exactly 0
            - Weakly neutral docs → slight lean towards pos or neg
        """
        probs = {item['label'].lower(): item['score'] for item in all_scores}
        p_pos = probs.get('positive', 0.0)
        p_neu = probs.get('neutral', 0.0)
        p_neg = probs.get('negative', 0.0)
        
        # Expected sentiment on [-1, +1]
        compound_score = p_pos - p_neg
        
        # Top label is the one with highest probability
        top_label = max(probs, key=probs.get)
        top_score = probs[top_label]
        
        return {
            'label': top_label,
            'score': top_score,
            'compound_score': float(compound_score),
            'prob_positive': float(p_pos),
            'prob_neutral': float(p_neu),
            'prob_negative': float(p_neg),
            'num_chunks': num_chunks
        }
    
    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """
        Analyze sentiment for multiple documents in batches
        
        Args:
            texts: List of document texts
            batch_size: Batch size for processing
        
        Returns:
            List of sentiment dictionaries (one per document)
        
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> texts = ["Good news!", "Bad news!", "Okay news."]
            >>> results = analyzer.analyze_batch(texts)
            >>> len(results)
            3
        """
        results = []
        total = len(texts)
        
        logger.info(f"Analyzing {total} documents...")
        
        for i, text in enumerate(texts):
            # Analyze document
            sentiment = self.analyze_document(text)
            results.append(sentiment)
            
            # Log progress
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{total} documents")
            
            # Clear GPU cache periodically
            if self.device >= 0 and (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
        
        logger.info(f"Completed sentiment analysis for {total} documents")
        
        return results
    
    def get_sentiment_distribution(self, sentiments: List[Dict]) -> Dict:
        """
        Get distribution statistics for a list of sentiments
        
        Args:
            sentiments: List of sentiment dictionaries
        
        Returns:
            Dictionary with distribution statistics
        """
        from .utils import calculate_sentiment_statistics
        return calculate_sentiment_statistics(sentiments)
