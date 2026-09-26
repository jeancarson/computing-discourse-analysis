"""
Script to run sentiment analysis on processed documents

USAGE:
    # Basic usage (auto-detects GPU/CPU)
    python run_sentiment_analysis.py --era 1
    
    # Force CPU only
    python run_sentiment_analysis.py --era 2 --device -1
    
    # Custom paths
    python run_sentiment_analysis.py --era 1 --data-root /path/to/data --output-dir /path/to/outputs
    
    # With merged topic groups (JSON format)
    python run_sentiment_analysis.py --era 1 --merged-topics '{"business": [0, 2, 7], "technical": [1, 3]}'

OUTPUTS (saved to outputs/era{N}/):
    - Era{N}_document_sentiments.csv     Document-level sentiments
    - Era{N}_topic_sentiments.csv        Topic-level aggregated sentiments
    - Era{N}_topic_sentiments_temporal.csv   Sentiment by topic over time
    - Era{N}_sentiment_summary.json      Summary statistics
    - Era{N}_sentiment_temporal.html     Interactive temporal line plot
    - Era{N}_sentiment_heatmap.html      Topic-time heatmap
    - Era{N}_sentiment_distribution.html Sentiment distribution by topic
    
    For merged topics:
    - era{N}_{group}_topic_sentiments.csv
    - era{N}_{group}_topic_sentiments_temporal.csv

REQUIREMENTS:
    - Topic modeling must be run first (needs Era{N}_document_topics.csv)
    - Preprocessed documents in data/processed/Era{N}/
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List

from .sentiment_analyzer import SentimentAnalyzer
from .aggregator import TopicSentimentAggregator
from .utils import calculate_sentiment_statistics, load_documents_for_sentiment
from .visualizations import create_all_visualizations

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_documents(processed_dir: Path, era: int) -> tuple[List[str], List[str], List[str], List[Dict]]:
    """
    Load preprocessed documents and their metadata.
    
    This is a wrapper around load_documents_for_sentiment that maintains
    backwards compatibility with the original function signature.
    
    Args:
        processed_dir: Path to processed data directory (parent of Era folders)
        era: Era number (1 or 2)
    
    Returns:
        Tuple of (texts, doc_ids, publication_dates, metadata)
    """
    # processed_dir is data/processed, so data_root is its parent
    data_root = processed_dir.parent
    return load_documents_for_sentiment(data_root, era)


def load_document_topics(output_dir: Path, era: int) -> pd.DataFrame:
    """
    Load document-topic assignments
    
    Args:
        output_dir: Path to output directory
        era: Era number (1 or 2)
    
    Returns:
        DataFrame with document-topic mappings
    """
    era_output = output_dir / f'era{era}'
    topics_file = era_output / f'Era{era}_document_topics.csv'
    
    if not topics_file.exists():
        raise FileNotFoundError(f"Document topics file not found: {topics_file}")
    
    doc_topics = pd.read_csv(topics_file)
    logger.info(f"Loaded topic assignments for {len(doc_topics)} documents")
    
    return doc_topics


def main(
    data_root: Path,
    output_dir: Path,
    era: int,
    device: int = None,
    batch_size: int = 8,
    merged_topics: Dict[str, List[int]] = None
):
    """
    Main execution function
    
    Args:
        data_root: Root directory for data
        output_dir: Output directory for results
        era: Era number (1 or 2)
        device: Device for inference (None=auto, -1=CPU, 0+=GPU)
        batch_size: Batch size for processing
        merged_topics: Optional dict of {group_name: [topic_ids]} for merged topic analysis
    """
    logger.info(f"Starting sentiment analysis for Era{era}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    era_output_dir = output_dir / f'era{era}'
    era_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load documents
    logger.info("Step 1: Loading preprocessed documents...")
    processed_dir = data_root / 'processed'
    texts, doc_ids, publication_dates, metadata = load_documents(processed_dir, era)
    
    # Step 2: Load topic assignments
    logger.info("Step 2: Loading document-topic assignments...")
    doc_topics = load_document_topics(output_dir, era)
    
    # Step 3: Analyze document sentiments
    logger.info("Step 3: Analyzing document-level sentiments...")
    analyzer = SentimentAnalyzer(device=device)
    sentiments = analyzer.analyze_batch(texts, batch_size=batch_size)
    
    # Step 4: Create document-level results
    logger.info("Step 4: Creating document-level sentiment results...")
    doc_sentiment_df = pd.DataFrame({
        'doc_id': doc_ids,
        'publication_date': publication_dates,
        'sentiment_label': [s['label'] for s in sentiments],
        'sentiment_score': [s['score'] for s in sentiments],
        'compound_score': [s.get('compound_score', None) for s in sentiments],
        'prob_positive': [s.get('prob_positive', None) for s in sentiments],
        'prob_neutral': [s.get('prob_neutral', None) for s in sentiments],
        'prob_negative': [s.get('prob_negative', None) for s in sentiments],
        'num_chunks_analyzed': [s.get('num_chunks', 1) for s in sentiments]
    })
    
    # Save document-level sentiments
    doc_sentiment_file = era_output_dir / f'Era{era}_document_sentiments.csv'
    doc_sentiment_df.to_csv(doc_sentiment_file, index=False)
    logger.info(f"Saved document sentiments to {doc_sentiment_file}")
    
    # Step 5: Aggregate by topic (overall)
    logger.info("Step 5: Aggregating sentiments by topic...")
    aggregator = TopicSentimentAggregator()
    topic_sentiments = aggregator.aggregate_by_topic(
        doc_topics,
        sentiments,
        doc_ids
    )
    
    # Add dominant sentiment category
    topic_sentiments = aggregator.get_dominant_sentiment_per_topic(topic_sentiments)
    
    # Save topic-level sentiments (overall)
    topic_sentiment_file = era_output_dir / f'Era{era}_topic_sentiments.csv'
    topic_sentiments.to_csv(topic_sentiment_file, index=False)
    logger.info(f"Saved topic sentiments to {topic_sentiment_file}")
    
    # Step 5b: Aggregate by topic AND time (temporal analysis)
    logger.info("Step 5b: Aggregating sentiments by topic over time...")
    topic_sentiments_temporal = aggregator.aggregate_by_topic_temporal(
        doc_topics,
        sentiments,
        doc_ids,
        publication_dates,
        time_period='year'
    )
    
    # Save temporal topic-level sentiments
    temporal_file = era_output_dir / f'Era{era}_topic_sentiments_temporal.csv'
    topic_sentiments_temporal.to_csv(temporal_file, index=False)
    logger.info(f"Saved temporal topic sentiments to {temporal_file}")
    
    # Step 6: Generate summary
    logger.info("Step 6: Generating summary statistics...")
    summary = aggregator.get_topic_sentiment_summary(topic_sentiments)
    
    # Add overall document statistics
    overall_stats = calculate_sentiment_statistics(sentiments)
    summary['overall_document_stats'] = overall_stats
    
    # Add temporal statistics
    summary['temporal_stats'] = {
        'num_time_periods': int(topic_sentiments_temporal['time_period'].nunique()),
        'time_range': {
            'start': str(topic_sentiments_temporal['time_period'].min()),
            'end': str(topic_sentiments_temporal['time_period'].max())
        },
        'topics_with_temporal_data': int(topic_sentiments_temporal['topic'].nunique())
    }
    
    # Save summary
    summary_file = era_output_dir / f'Era{era}_sentiment_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SENTIMENT ANALYSIS SUMMARY - ERA {era}")
    print(f"{'='*70}")
    print(f"Total documents analyzed: {len(doc_ids)}")
    print(f"Total topics: {summary['num_topics']}")
    print(f"\nOverall document sentiment:")
    print(f"  Mean: {overall_stats['mean_sentiment']:.3f}")
    print(f"  Positive: {overall_stats['positive_pct']*100:.1f}%")
    print(f"  Neutral: {overall_stats['neutral_pct']*100:.1f}%")
    print(f"  Negative: {overall_stats['negative_pct']*100:.1f}%")
    print(f"\nTopic-level sentiment (overall):")
    print(f"  Mean across topics: {summary['mean_sentiment_across_topics']:.3f}")
    print(f"  Most positive topic: {summary['most_positive_topic']} ({summary['most_positive_score']:.3f})")
    print(f"  Most negative topic: {summary['most_negative_topic']} ({summary['most_negative_score']:.3f})")
    print(f"\nTemporal analysis:")
    print(f"  Time periods: {summary['temporal_stats']['num_time_periods']}")
    print(f"  Time range: {summary['temporal_stats']['time_range']['start']} to {summary['temporal_stats']['time_range']['end']}")
    print(f"  Topics tracked: {summary['temporal_stats']['topics_with_temporal_data']}")
    print(f"{'='*70}\n")
    
    # Step 7: Process merged topics if provided
    if merged_topics:
        logger.info(f"Step 7: Processing {len(merged_topics)} merged topic groups...")
        
        for group_name, topic_ids in merged_topics.items():
            print(f"\nProcessing merged topic '{group_name}': {topic_ids}")
            
            # Filter doc_topics to only include specified topics
            merged_doc_topics = doc_topics[doc_topics['topic'].isin(topic_ids)].copy()
            
            if len(merged_doc_topics) == 0:
                logger.warning(f"No documents found for topics {topic_ids}")
                continue
            
            # Aggregate sentiment for merged topics (overall)
            merged_sentiments = aggregator.aggregate_by_topic(merged_doc_topics, sentiments, doc_ids)
            merged_file = era_output_dir / f'era{era}_{group_name}_topic_sentiments.csv'
            merged_sentiments.to_csv(merged_file, index=False)
            
            # Aggregate temporal sentiments for merged topics
            merged_temporal = aggregator.aggregate_by_topic_temporal(
                merged_doc_topics, sentiments, doc_ids, publication_dates, time_period='year'
            )
            temporal_merged_file = era_output_dir / f'era{era}_{group_name}_topic_sentiments_temporal.csv'
            merged_temporal.to_csv(temporal_merged_file, index=False)
            
            print(f"  ✓ Saved {merged_file.name}")
            print(f"  ✓ Saved {temporal_merged_file.name}")
    
    # Step 8: Generate visualizations
    logger.info("Step 8: Generating visualizations...")
    try:
        create_all_visualizations(era, era_output_dir, top_n_topics=10)
        print(f"✓ Generated interactive visualizations")
    except Exception as e:
        logger.warning(f"Failed to generate visualizations: {e}")
    
    logger.info(f"Sentiment analysis complete for Era{era}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run sentiment analysis on topic modeling results'
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=Path('../../../data'),
        help='Root directory for data'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('../../outputs'),
        help='Output directory'
    )
    parser.add_argument(
        '--era',
        type=int,
        required=True,
        choices=[1, 2],
        help='Era to process (1 or 2)'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=None,
        help='Device for inference (None=auto, -1=CPU, 0+=GPU)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--merged-topics',
        type=str,
        default=None,
        help='JSON string of merged topics: {"group_name": [topic_ids]}'
    )
    
    args = parser.parse_args()
    
    # Parse merged topics if provided
    merged_topics = None
    if args.merged_topics:
        import json
        merged_topics = json.loads(args.merged_topics)
    
    main(
        data_root=args.data_root,
        output_dir=args.output_dir,
        era=args.era,
        device=args.device,
        batch_size=args.batch_size,
        merged_topics=merged_topics
    )
