"""
Sentiment Analysis Module

This module provides sentiment analysis functionality for documents and topics,
using RoBERTa-base model fine-tuned on Twitter sentiment data.

Features:
- Document-level sentiment analysis with RoBERTa
- Topic-level sentiment aggregation (overall and temporal)
- Interactive visualizations for temporal trends
- Support for long documents via chunking
"""

from .sentiment_analyzer import SentimentAnalyzer
from .aggregator import TopicSentimentAggregator
from .utils import convert_sentiment_to_numeric, chunk_text_by_sentences, load_documents_for_sentiment

__all__ = [
    'SentimentAnalyzer',
    'TopicSentimentAggregator',
    'convert_sentiment_to_numeric',
    'chunk_text_by_sentences',
    'load_documents_for_sentiment'
]
