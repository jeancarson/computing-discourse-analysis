"""
Topic-level sentiment aggregation
"""

import logging
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

from .utils import convert_sentiment_to_numeric, calculate_sentiment_statistics

logger = logging.getLogger(__name__)


class TopicSentimentAggregator:
    """
    Aggregates document-level sentiments to topic-level statistics
    
    Following Xian et al. (2024):
    - Overall sentiment masks important variations across topics
    - Different topics evoke different emotional responses
    - Topic-specific sentiment reveals nuanced public reception
    
    Example:
        Business topics: more positive (0.3-0.5)
        Regulation topics: more neutral/negative (-0.1 to 0.1)
    """
    
    def __init__(self):
        """Initialize aggregator"""
        pass
    
    def aggregate_by_topic(
        self,
        document_topics: pd.DataFrame,
        document_sentiments: List[Dict],
        doc_ids: List[str]
    ) -> pd.DataFrame:
        """
        Aggregate document sentiments to topic-level statistics
        
        Args:
            document_topics: DataFrame with columns ['doc_id', 'topic', 'probability']
            document_sentiments: List of sentiment dicts (same order as doc_ids)
            doc_ids: List of document IDs (same order as sentiments)
        
        Returns:
            DataFrame with topic-level sentiment statistics
        
        Columns:
            - topic: Topic ID
            - mean_sentiment: Mean numeric sentiment (-1 to +1)
            - std_sentiment: Standard deviation of sentiment
            - positive_pct: Percentage of positive documents
            - neutral_pct: Percentage of neutral documents
            - negative_pct: Percentage of negative documents
            - doc_count: Number of documents in topic
        """
        logger.info("Aggregating sentiments by topic...")
        
        # Create document-level dataframe
        doc_sentiment_df = pd.DataFrame({
            'doc_id': doc_ids,
            'sentiment_label': [s['label'] for s in document_sentiments],
            'sentiment_score': [s['score'] for s in document_sentiments],
            'compound_score': [s.get('compound_score', None) for s in document_sentiments],
            'numeric_sentiment': [
                convert_sentiment_to_numeric(
                    s['label'], s['score'],
                    compound_score=s.get('compound_score', None)
                )
                for s in document_sentiments
            ]
        })
        
        # Merge with topics
        merged = document_topics.merge(doc_sentiment_df, on='doc_id', how='left')
        
        # Group by topic and calculate statistics
        topic_stats = []
        
        for topic_id in sorted(merged['topic'].unique()):
            topic_docs = merged[merged['topic'] == topic_id]
            
            if len(topic_docs) == 0:
                continue
            
            # Get sentiments for this topic
            topic_sentiments = [
                {
                    'label': row['sentiment_label'],
                    'score': row['sentiment_score'],
                    'compound_score': row['compound_score'] if pd.notna(row.get('compound_score')) else None
                }
                for _, row in topic_docs.iterrows()
                if pd.notna(row['sentiment_label'])
            ]
            
            # Calculate statistics
            stats = calculate_sentiment_statistics(topic_sentiments)
            stats['topic'] = topic_id
            
            topic_stats.append(stats)
        
        # Create results dataframe
        results_df = pd.DataFrame(topic_stats)
        
        # Reorder columns
        column_order = [
            'topic',
            'mean_sentiment',
            'std_sentiment',
            'positive_pct',
            'neutral_pct',
            'negative_pct',
            'count'
        ]
        results_df = results_df[column_order]
        results_df.rename(columns={'count': 'doc_count'}, inplace=True)
        
        # Sort by topic ID
        results_df = results_df.sort_values('topic').reset_index(drop=True)
        
        logger.info(f"Aggregated sentiments for {len(results_df)} topics")
        
        return results_df
    
    def get_topic_sentiment_summary(self, topic_sentiments: pd.DataFrame) -> Dict:
        """
        Get summary statistics across all topics
        
        Args:
            topic_sentiments: DataFrame from aggregate_by_topic()
        
        Returns:
            Dictionary with overall statistics
        """
        return {
            'num_topics': len(topic_sentiments),
            'mean_sentiment_across_topics': float(topic_sentiments['mean_sentiment'].mean()),
            'std_sentiment_across_topics': float(topic_sentiments['mean_sentiment'].std()),
            'most_positive_topic': int(topic_sentiments.loc[topic_sentiments['mean_sentiment'].idxmax(), 'topic']),
            'most_negative_topic': int(topic_sentiments.loc[topic_sentiments['mean_sentiment'].idxmin(), 'topic']),
            'most_positive_score': float(topic_sentiments['mean_sentiment'].max()),
            'most_negative_score': float(topic_sentiments['mean_sentiment'].min())
        }
    
    def filter_by_sentiment(
        self,
        topic_sentiments: pd.DataFrame,
        min_sentiment: float = None,
        max_sentiment: float = None
    ) -> pd.DataFrame:
        """
        Filter topics by sentiment score range
        
        Args:
            topic_sentiments: DataFrame from aggregate_by_topic()
            min_sentiment: Minimum sentiment score (inclusive)
            max_sentiment: Maximum sentiment score (inclusive)
        
        Returns:
            Filtered DataFrame
        
        Example:
            # Get only positive topics
            >>> positive_topics = aggregator.filter_by_sentiment(
            ...     topic_sentiments, min_sentiment=0.2
            ... )
        """
        filtered = topic_sentiments.copy()
        
        if min_sentiment is not None:
            filtered = filtered[filtered['mean_sentiment'] >= min_sentiment]
        
        if max_sentiment is not None:
            filtered = filtered[filtered['mean_sentiment'] <= max_sentiment]
        
        return filtered.reset_index(drop=True)
    
    def get_dominant_sentiment_per_topic(
        self,
        topic_sentiments: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Determine dominant sentiment category for each topic
        
        Args:
            topic_sentiments: DataFrame from aggregate_by_topic()
        
        Returns:
            DataFrame with additional 'dominant_sentiment' column
        """
        df = topic_sentiments.copy()
        
        # Determine dominant sentiment based on highest percentage
        def get_dominant(row):
            pcts = {
                'positive': row['positive_pct'],
                'neutral': row['neutral_pct'],
                'negative': row['negative_pct']
            }
            return max(pcts, key=pcts.get)
        
        df['dominant_sentiment'] = df.apply(get_dominant, axis=1)
        
        return df
    
    def aggregate_by_topic_temporal(
        self,
        document_topics: pd.DataFrame,
        document_sentiments: List[Dict],
        doc_ids: List[str],
        publication_dates: List[str],
        time_period: str = 'year'
    ) -> pd.DataFrame:
        """
        Aggregate document sentiments by topic AND time period
        
        This enables temporal analysis to track how sentiment changes
        over time for each topic.
        
        Args:
            document_topics: DataFrame with columns ['doc_id', 'topic', 'probability']
            document_sentiments: List of sentiment dicts (same order as doc_ids)
            doc_ids: List of document IDs (same order as sentiments)
            publication_dates: List of publication dates (ISO format, same order)
            time_period: Aggregation period ('year', 'quarter', 'month')
        
        Returns:
            DataFrame with temporal topic-level sentiment statistics
        
        Columns:
            - topic: Topic ID
            - time_period: Time period (e.g., '1972', '1972-Q1', '1972-01')
            - mean_sentiment: Mean numeric sentiment
            - std_sentiment: Standard deviation
            - positive_pct, neutral_pct, negative_pct: Distribution
            - doc_count: Number of documents
        """
        logger.info(f"Aggregating sentiments by topic and {time_period}...")
        
        # Create document-level dataframe with dates
        doc_sentiment_df = pd.DataFrame({
            'doc_id': doc_ids,
            'sentiment_label': [s['label'] for s in document_sentiments],
            'sentiment_score': [s['score'] for s in document_sentiments],
            'compound_score': [s.get('compound_score', None) for s in document_sentiments],
            'publication_date': publication_dates,
            'numeric_sentiment': [
                convert_sentiment_to_numeric(
                    s['label'], s['score'],
                    compound_score=s.get('compound_score', None)
                )
                for s in document_sentiments
            ]
        })
        
        # Parse dates
        doc_sentiment_df['publication_date'] = pd.to_datetime(
            doc_sentiment_df['publication_date'],
            errors='coerce'
        )
        
        # Create time period column
        if time_period == 'year':
            doc_sentiment_df['time_period'] = doc_sentiment_df['publication_date'].dt.year.astype(str)
        elif time_period == 'quarter':
            doc_sentiment_df['time_period'] = (
                doc_sentiment_df['publication_date'].dt.year.astype(str) + '-Q' +
                doc_sentiment_df['publication_date'].dt.quarter.astype(str)
            )
        elif time_period == 'month':
            doc_sentiment_df['time_period'] = doc_sentiment_df['publication_date'].dt.to_period('M').astype(str)
        else:
            raise ValueError(f"Invalid time_period: {time_period}. Use 'year', 'quarter', or 'month'")
        
        # Merge with topics
        merged = document_topics.merge(doc_sentiment_df, on='doc_id', how='left')
        
        # Drop rows with missing dates
        merged = merged.dropna(subset=['time_period'])
        
        # Group by topic and time period
        temporal_stats = []
        
        for (topic_id, period) in merged.groupby(['topic', 'time_period']).groups.keys():
            topic_period_docs = merged[
                (merged['topic'] == topic_id) & 
                (merged['time_period'] == period)
            ]
            
            if len(topic_period_docs) == 0:
                continue
            
            # Get sentiments for this topic-period combination
            sentiments = [
                {
                    'label': row['sentiment_label'],
                    'score': row['sentiment_score'],
                    'compound_score': row['compound_score'] if pd.notna(row.get('compound_score')) else None
                }
                for _, row in topic_period_docs.iterrows()
                if pd.notna(row['sentiment_label'])
            ]
            
            # Calculate statistics
            stats = calculate_sentiment_statistics(sentiments)
            stats['topic'] = topic_id
            stats['time_period'] = period
            
            temporal_stats.append(stats)
        
        # Create results dataframe
        results_df = pd.DataFrame(temporal_stats)
        
        # Reorder columns
        column_order = [
            'topic',
            'time_period',
            'mean_sentiment',
            'std_sentiment',
            'positive_pct',
            'neutral_pct',
            'negative_pct',
            'count'
        ]
        results_df = results_df[column_order]
        results_df.rename(columns={'count': 'doc_count'}, inplace=True)
        
        # Sort by topic and time period
        results_df = results_df.sort_values(['topic', 'time_period']).reset_index(drop=True)
        
        logger.info(
            f"Aggregated sentiments for {results_df['topic'].nunique()} topics "
            f"across {results_df['time_period'].nunique()} time periods"
        )
        
        return results_df
