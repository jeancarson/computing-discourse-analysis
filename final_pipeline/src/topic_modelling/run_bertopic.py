"""
BERTopic Topic modelling Pipeline for Computing Discourse Eras

Usage:
    python run_bertopic.py                          # Process both eras, all years
    python run_bertopic.py --era 1                  # Process only Era1
    python run_bertopic.py --years 1970             # Process only 1970
    python run_bertopic.py --years 1970 1975        # Process 1970-1975
    python run_bertopic.py --era 2 --years 2023 2025  # Process Era2, 2023-2025
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import sys

import pandas as pd

# Add src directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from topic_modelling.core.topic_modeler import TopicModeler
from topic_modelling.utils.data_utils import load_processed_data
from topic_modelling.utils.chunker import SemanticChunker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def log_pipeline_run(log_path: Path, era: str, stats: Dict):
    """Append pipeline run statistics to log file"""
    timestamp = datetime.now().isoformat()

    year_filter_info = ""
    if 'year_filter' in stats:
        year_filter_info = f"\nYear Filter: {stats['year_filter']}"

    log_entry = f"""
{'='*80}
Pipeline Run: {timestamp}
Era: {era}{year_filter_info}
{'='*80}

Statistics:
- Documents processed: {stats.get('documents_processed', 0)}
- Topics identified: {stats.get('topics_identified', 0)}
- Outlier documents: {stats.get('outlier_documents', 0)}
- Processing time: {stats.get('processing_time', 'N/A')}

Configuration:
- Embedding model: {stats.get('embedding_model', 'N/A')}
- UMAP neighbors: {stats.get('umap_neighbors', 'N/A')}
- HDBSCAN min_cluster_size: {stats.get('hdbscan_min_cluster_size', 'N/A')}

{'='*80}

"""

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_entry)


def calculate_multi_topic_info(bertopic_model, topics, probs, threshold: float) -> pd.DataFrame:
    """
    Calculate topic info with multi-topic assignments.
    Documents can belong to multiple topics if probability > threshold.
    """
    import numpy as np
    
    primary_topic_info = bertopic_model.get_topic_info()
    
    num_topics = probs.shape[1]
    topic_counts = {}
    
    # Count how many documents have prob > threshold for each topic
    for topic_id in range(num_topics):
        # Get probability column for this topic
        topic_probs = probs[:, topic_id]
        count = int(np.sum(topic_probs >= threshold))
        topic_counts[topic_id] = count
    
    # Create topic info dataframe
    topic_info_data = []
    for topic_id, count in topic_counts.items():
        if count > 0:  # Only include topics with at least one assignment
            # Find matching row in primary topic_info
            original_row = primary_topic_info[primary_topic_info['Topic'] == topic_id]
            
            if not original_row.empty:
                topic_info_data.append({
                    'Topic': topic_id,
                    'Count': count,
                    'Name': original_row.iloc[0]['Name'],
                    'Representation': original_row.iloc[0]['Representation'],
                    'Representative_Docs': original_row.iloc[0]['Representative_Docs']
                })
            else:
                topic_info_data.append({
                    'Topic': topic_id,
                    'Count': count,
                    'Name': f'Topic_{topic_id}',
                    'Representation': '',
                    'Representative_Docs': ''
                })
    
    return pd.DataFrame(topic_info_data).sort_values('Count', ascending=False).reset_index(drop=True)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BERTopic Topic modelling Pipeline')
    parser.add_argument('--era', type=int, choices=[1, 2], help='Process only era 1 or 2')
    parser.add_argument('--years', type=int, nargs='+', metavar='YEAR', help='Filter by year(s)')
    
    args = parser.parse_args()
    if args.years and len(args.years) > 2:
        parser.error("--years accepts at most 2 values")
    
    return args


def run_bertopic_modelling(processed_texts: list, metadata_list: list, config, output_dir: Path, era: str, year_filter=None):
    """
    Run BERTopic modelling on preprocessed texts.
    This function is independent and can be called from notebooks or other scripts.
    """
    start_time = datetime.now()
    
    # Initialize modeler
    modeler = TopicModeler(config)
    
    # Model topics
    topics, probs = modeler.fit_transform(processed_texts)
    topic_keywords = modeler.get_topic_keywords()
    
    # Handle multi-topic assignment if enabled
    if config.multi_topic_assignment.enabled and probs is not None:
        logger.info(f"Multi-topic assignment enabled (threshold: {config.multi_topic_assignment.probability_threshold})")
        topic_info = calculate_multi_topic_info(
            modeler.model, topics, probs, config.multi_topic_assignment.probability_threshold
        )
    else:
        topic_info = modeler.get_topic_info()

    # Determine output path
    if year_filter:
        era_output_dir = output_dir / "years" / year_filter
        file_prefix = f"{era}_{year_filter}"
    else:
        era_output_dir = output_dir / era.lower()
        file_prefix = era
    
    era_output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    topic_info.to_csv(era_output_dir / f"{file_prefix}_topic_info.csv", index=False)
    topic_keywords.to_csv(era_output_dir / f"{file_prefix}_topic_keywords.csv", index=False)
    
    # Handle assignments
    if config.multi_topic_assignment.enabled and probs is not None:
        assignments_list = []
        metadata_df = pd.DataFrame(metadata_list)
        threshold = config.multi_topic_assignment.probability_threshold
        
        for doc_idx in range(len(topics)):
            doc_probs = probs[doc_idx]
            assigned_topics = [(topic_id, prob) for topic_id, prob in enumerate(doc_probs) 
                             if prob >= threshold]
            
            for topic_id, prob in assigned_topics:
                assignment_row = {
                    'document_index': doc_idx,
                    'topic': topic_id,
                    'probability': prob,
                    **metadata_df.iloc[doc_idx].to_dict()
                }
                assignments_list.append(assignment_row)
        
        assignments_df = pd.DataFrame(assignments_list)
    else:
        assignments_df = pd.DataFrame({
            'document_index': range(len(topics)),
            'topic': topics,
            'probability': probs.max(axis=1) if probs is not None else [0] * len(topics),
            **pd.DataFrame(metadata_list).to_dict('series')
        })
    
    assignments_df.to_csv(era_output_dir / f"{file_prefix}_topic_assignments.csv", index=False)
    
    if probs is not None:
        probs_df = pd.DataFrame(
            probs,
            columns=[f"topic_{i}_prob" for i in range(probs.shape[1])]
        )
        probs_df.insert(0, 'document_index', range(len(topics)))
        probs_df.to_csv(era_output_dir / f"{file_prefix}_topic_probabilities.csv", index=False)
    
    modeler.save_visualizations(era_output_dir, file_prefix)

    # Log statistics
    stats = {
        'documents_processed': len(processed_texts),
        'topics_identified': len(topic_info[topic_info['Topic'] != -1]),
        'outlier_documents': int(topic_info[topic_info['Topic'] == -1]['Count'].sum()) if -1 in topic_info['Topic'].values else 0,
        'processing_time': str(datetime.now() - start_time),
        'embedding_model': config.embedding_model,
        'umap_neighbors': config.umap.n_neighbors,
        'hdbscan_min_cluster_size': config.hdbscan.min_cluster_size
    }
    if year_filter:
        stats['year_filter'] = year_filter

    log_pipeline_run(Path(__file__).parent / "pipeline.log", era, stats)
    logger.info(f"Completed {era}. Results saved to: {era_output_dir}")
    
    return topic_info, assignments_df, topic_keywords


def main():
    """Main pipeline execution"""
    args = parse_arguments()
    
    script_dir = Path(__file__).parent
    data_root = script_dir.parent.parent.parent / "data"
    output_dir = script_dir.parent.parent / "outputs"  # Centralized output folder
    
    year_start, year_end = None, None
    if args.years:
        year_start = min(args.years)
        year_end = max(args.years) if len(args.years) == 2 else None
    
    eras = [args.era] if args.era else [1, 2]
    
    for era in eras:
        logger.info(f"Starting Era{era}" + (f" ({year_start}-{year_end or year_start})" if year_start else ""))
        start_time = datetime.now()

        try:
            # Load era-specific configuration from centralized config
            config = load_config(era=era)
            logger.info(f"Loaded configuration for Era{era}")
            
            texts, metadata_list = load_processed_data(data_root, era)
            
            if not texts:
                logger.warning(f"No documents found for Era{era}. Skipping.")
                continue

            chunker = SemanticChunker()
            chunked_texts, chunked_metadata = chunker.chunk_documents(texts, metadata_list)

            year_filter = f"{year_start}-{year_end}" if year_end else str(year_start) if year_start else None
            run_bertopic_modelling(chunked_texts, chunked_metadata, config, output_dir, f'Era{era}', year_filter)

        except Exception as e:
            logger.error(f"Error processing Era{era}: {e}", exc_info=True)

    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()