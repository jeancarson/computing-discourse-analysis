"""
Run Cross-Era Topic Similarity Analysis

Standalone script to generate 3 topic similarity heatmaps:
1. Era 1 within-era topic similarity
2. Era 2 within-era topic similarity
3. Cross-era topic similarity (Era 1 vs Era 2)

Requires:
- Cached embeddings (from main pipeline)
- Topic assignment CSVs (from main pipeline)

Usage:
    python run_similarity_analysis.py
    python run_similarity_analysis.py --top-n 10
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from cross_era_analysis.similarity_analyzer import EmbeddingSimilarityAnalyzer
from cross_era_analysis.visualizations import SimilarityVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Cross-Era Topic Similarity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of top matches to display per topic (default: 5)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory (default: outputs/cross_era_analysis)'
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_root = project_root / "data"
    outputs_dir = script_dir.parent / "outputs"
    
    embeddings_cache_dir = outputs_dir / "embeddings_cache"
    output_dir = Path(args.output_dir) if args.output_dir else outputs_dir / "cross_era_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("CROSS-ERA TOPIC SIMILARITY ANALYSIS")
    logger.info("="*70)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Embeddings cache: {embeddings_cache_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Get embedding model name from config
    config = load_config(era=1)
    embedding_model_name = config.embedding_model
    logger.info(f"Embedding model: {embedding_model_name}")
    
    # Initialize analyzer
    analyzer = EmbeddingSimilarityAnalyzer(
        embeddings_cache_dir=embeddings_cache_dir,
        embedding_model_name=embedding_model_name
    )
    
    # Load data
    try:
        logger.info("\nLoading embeddings and metadata...")
        analyzer.load_embeddings(era=1)
        analyzer.load_embeddings(era=2)
        analyzer.load_metadata(data_root, era=1)
        analyzer.load_metadata(data_root, era=2)
        
    except FileNotFoundError as e:
        logger.error(f"\n{e}")
        logger.error("\nPlease run the main pipeline first to generate embeddings.")
        sys.exit(1)
    
    # Generate heatmaps
    logger.info("\n" + "="*70)
    logger.info("GENERATING TOPIC SIMILARITY HEATMAPS")
    logger.info("="*70)
    
    visualizer = SimilarityVisualizer(output_dir)
    visualizer.create_all_heatmaps(analyzer, outputs_dir, top_n_matches=args.top_n)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)
    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info("  • era1_within_similarity.html")
    logger.info("  • era2_within_similarity.html")
    logger.info("  • cross_era_similarity.html")
    logger.info("  • topic_similarity.csv")


if __name__ == "__main__":
    main()
