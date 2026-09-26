"""
Script to extract and rank documents by their influence/relevance to specific topics.

This script reads the topic probabilities and assignments from BERTopic outputs
and ranks documents by their probability scores for each topic.

Usage:
    python get_top_docs_by_topic.py <era> <topic_number> [--top N]
    
Examples:
    python get_top_docs_by_topic.py era1 5                    # Get all docs for topic 5
    python get_top_docs_by_topic.py era1 5 --top 10           # Get top 10 docs for topic 5
    python get_top_docs_by_topic.py era2 12 --top 20          # Get top 20 docs for topic 12
    
Output:
    - Displays documents ranked by probability (most influential first)
    - Shows document name and probability score
    - Optional: limit to top N documents
"""

import sys
import argparse
import pandas as pd
from pathlib import Path


def get_top_documents_for_topic(era: str, topic_num: int, top_n: int = None, output_dir: Path = None):
    """
    Get documents ranked by their relevance to a specific topic.
    
    Args:
        era: Era identifier (e.g., 'era1', 'era2')
        topic_num: Topic number to query
        top_n: Optional limit on number of documents to return
        output_dir: Optional custom output directory path
        
    Returns:
        DataFrame with ranked documents and their probabilities
    """
    era = era.lower()
    era_formatted = era.capitalize()
    
    # Determine output directory
    if output_dir is None:
        # Default: relative to script location
        script_dir = Path(__file__).parent.parent.parent.parent
        output_dir = script_dir / "outputs"
    
    # Load document topics file
    doc_topics_file = output_dir / era / f"{era_formatted}_document_topics.csv"
    
    if not doc_topics_file.exists():
        raise FileNotFoundError(f"Document topics file not found: {doc_topics_file}")
    
    # Load data
    df = pd.read_csv(doc_topics_file)
    
    # Check if topic exists
    available_topics = sorted(df['topic'].unique())
    if topic_num not in available_topics:
        raise ValueError(f"Topic {topic_num} not found. Available topics: {', '.join(map(str, available_topics))}")
    
    # Filter for the specified topic
    result_df = df[df['topic'] == topic_num].copy()
    
    # Sort by probability (descending)
    result_df = result_df.sort_values('probability', ascending=False)
    
    # Rename doc_id to filepath for consistency
    result_df = result_df.rename(columns={'doc_id': 'filepath'})
    
    # Select relevant columns
    result_df = result_df[['filepath', 'probability', 'chunk_count', 'num_topics']]
    
    # Limit to top N if specified
    if top_n:
        result_df = result_df.head(top_n)
    
    return result_df


def main():
    """Main execution for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Extract and rank documents by topic relevance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s era1 5              Get all documents for Era1, Topic 5
  %(prog)s era1 5 --top 10     Get top 10 documents for Era1, Topic 5
  %(prog)s era2 12 --top 20    Get top 20 documents for Era2, Topic 12
        """
    )
    
    parser.add_argument('era', type=str, help='Era identifier (era1, era2)')
    parser.add_argument('topic', type=int, help='Topic number to query')
    parser.add_argument('--top', type=int, metavar='N', 
                       help='Limit to top N documents (default: show all)')
    parser.add_argument('--output-dir', type=str, metavar='PATH',
                       help='Custom output directory path (default: ../../outputs)')
    
    args = parser.parse_args()
    
    try:
        output_dir = Path(args.output_dir) if args.output_dir else None
        result_df = get_top_documents_for_topic(
            args.era, 
            args.topic, 
            args.top,
            output_dir
        )
        
        # Display results
        era_display = args.era.upper()
        limit_text = f" (Top {args.top})" if args.top else ""
        print(f"\n{era_display} - Topic {args.topic}{limit_text}")
        print(f"Total Documents: {len(result_df)}")
        print("=" * 80)
        print()
        
        # Print table
        for idx, row in result_df.iterrows():
            multi_topic_marker = f" [{row['num_topics']} topics]" if row['num_topics'] > 1 else ""
            print(f"{row['probability']:.4f}  {row['filepath']}{multi_topic_marker}")
        
        print()
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
