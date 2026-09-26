"""
Visualization utilities for cross-era similarity analysis.

Generates 3 key topic similarity heatmaps:
1. Era 1 within-era topic similarity
2. Era 2 within-era topic similarity
3. Cross-era topic similarity (Era 1 vs Era 2)
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SimilarityVisualizer:
    """Create topic similarity heatmaps for cross-era analysis."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_topic_heatmap(
        self,
        similarity_matrix: np.ndarray,
        row_topic_ids: list,
        col_topic_ids: list,
        row_keywords: dict,
        col_keywords: dict,
        title: str,
        save_name: str,
        row_prefix: str = "T",
        col_prefix: str = "T"
    ) -> go.Figure:
        """
        Create an interactive topic similarity heatmap.
        
        Args:
            similarity_matrix: Topic similarity matrix
            row_topic_ids: Topic IDs for rows
            col_topic_ids: Topic IDs for columns
            row_keywords: Dict mapping topic_id -> keywords for rows
            col_keywords: Dict mapping topic_id -> keywords for columns
            title: Plot title
            save_name: Filename for saving
            row_prefix: Prefix for row labels
            col_prefix: Prefix for column labels
        
        Returns:
            Plotly figure
        """
        # Build labels with top 3 keywords
        row_labels = [
            f"{row_prefix}{tid}: {', '.join(row_keywords.get(tid, [])[:3])}"
            for tid in row_topic_ids
        ]
        col_labels = [
            f"{col_prefix}{tid}: {', '.join(col_keywords.get(tid, [])[:3])}"
            for tid in col_topic_ids
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=col_labels,
            y=row_labels,
            colorscale='RdYlGn',
            zmid=0.5,
            text=np.round(similarity_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Similarity"),
            hovertemplate='%{y}<br>%{x}<br>Similarity: %{z:.3f}<extra></extra>'
        ))
        
        n_rows, n_cols = len(row_topic_ids), len(col_topic_ids)
        fig.update_layout(
            title=f"{title}<br><sub>{n_rows} × {n_cols} topics</sub>",
            xaxis_title="Topics",
            yaxis_title="Topics",
            width=max(800, n_cols * 50),
            height=max(600, n_rows * 50),
            xaxis={'side': 'bottom', 'tickangle': -45},
            yaxis={'autorange': 'reversed'}
        )
        
        save_path = self.output_dir / save_name
        fig.write_html(str(save_path))
        logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_all_heatmaps(self, analyzer, output_dir: Path, top_n_matches: int = 5):
        """
        Generate all 3 topic similarity heatmaps.
        
        This is the main entry point. Computes topic centroids from CSVs,
        then generates within-era and cross-era heatmaps.
        
        Args:
            analyzer: EmbeddingSimilarityAnalyzer with embeddings/metadata loaded
            output_dir: Directory containing era1/ and era2/ CSV files
            top_n_matches: Number of top matches to print per topic
        """
        # Compute centroids from CSV (no need for full pipeline)
        analyzer.compute_topic_centroids_from_csv(output_dir)
        
        era1_ids = analyzer.topic_ids[1]
        era2_ids = analyzer.topic_ids[2]
        era1_kw = {tid: analyzer.get_topic_keywords(1, tid) for tid in era1_ids}
        era2_kw = {tid: analyzer.get_topic_keywords(2, tid) for tid in era2_ids}
        
        era1_centroids = np.vstack([analyzer.topic_centroids[1][tid] for tid in era1_ids])
        era2_centroids = np.vstack([analyzer.topic_centroids[2][tid] for tid in era2_ids])
        
        # 1. Era 1 within-era
        era1_sim = cosine_similarity(era1_centroids, era1_centroids)
        self._create_topic_heatmap(
            era1_sim, era1_ids, era1_ids, era1_kw, era1_kw,
            title="Era 1: Within-Era Topic Similarity",
            save_name="era1_within_similarity.html",
            row_prefix="E1-T", col_prefix="E1-T"
        )
        print(f"Era 1 within-era: min={era1_sim.min():.3f}, max={era1_sim.max():.3f}")
        
        # Save Era 1 within-era similarity matrix as CSV
        self._save_similarity_matrix_csv(era1_sim, era1_ids, era1_ids, era1_kw, era1_kw, 
                                         'era1_within_similarity_matrix.csv', 'E1-T', 'E1-T')
        
        # 2. Era 2 within-era
        era2_sim = cosine_similarity(era2_centroids, era2_centroids)
        self._create_topic_heatmap(
            era2_sim, era2_ids, era2_ids, era2_kw, era2_kw,
            title="Era 2: Within-Era Topic Similarity",
            save_name="era2_within_similarity.html",
            row_prefix="E2-T", col_prefix="E2-T"
        )
        print(f"Era 2 within-era: min={era2_sim.min():.3f}, max={era2_sim.max():.3f}")
        
        # Save Era 2 within-era similarity matrix as CSV
        self._save_similarity_matrix_csv(era2_sim, era2_ids, era2_ids, era2_kw, era2_kw,
                                         'era2_within_similarity_matrix.csv', 'E2-T', 'E2-T')
        
        # 3. Cross-era
        cross_sim = cosine_similarity(era1_centroids, era2_centroids)
        self._create_topic_heatmap(
            cross_sim, era1_ids, era2_ids, era1_kw, era2_kw,
            title="Cross-Era Topic Similarity (Era 1 vs Era 2)",
            save_name="cross_era_similarity.html",
            row_prefix="E1-T", col_prefix="E2-T"
        )
        print(f"Cross-era: min={cross_sim.min():.3f}, max={cross_sim.max():.3f}")
        
        # Save cross-era similarity matrix as CSV
        self._save_similarity_matrix_csv(cross_sim, era1_ids, era2_ids, era1_kw, era2_kw,
                                         'era1_era2_cross_similarity_matrix.csv', 'E1-T', 'E2-T')
        
        # Save cross-era similarity to CSV (legacy format - list of all pairs)
        self._save_cross_era_csv(cross_sim, era1_ids, era2_ids, era1_kw, era2_kw)
        
        # Print top matches
        self._print_top_matches(cross_sim, era1_ids, era2_ids, era1_kw, era2_kw, top_n_matches)
        
        print(f"\n✓ All heatmaps and similarity matrices saved to {self.output_dir}")
    
    def _save_similarity_matrix_csv(
        self,
        similarity_matrix: np.ndarray,
        row_topic_ids: list,
        col_topic_ids: list,
        row_keywords: dict,
        col_keywords: dict,
        filename: str,
        row_prefix: str = "T",
        col_prefix: str = "T"
    ):
        """
        Save similarity matrix as CSV with topic labels.
        
        Creates a matrix CSV where:
        - Rows are indexed by topic IDs with keywords
        - Columns are topic IDs with keywords
        - Values are cosine similarities
        """
        # Create column labels with keywords
        col_labels = [f"{col_prefix}{tid}" for tid in col_topic_ids]
        
        # Create row labels with keywords
        row_labels = [f"{row_prefix}{tid}: {', '.join(row_keywords.get(tid, [])[:3])}" 
                      for tid in row_topic_ids]
        
        # Create DataFrame
        df = pd.DataFrame(
            similarity_matrix,
            index=row_labels,
            columns=col_labels
        )
        
        # Save to CSV
        csv_path = self.output_dir / filename
        df.to_csv(csv_path)
        logger.info(f"Saved similarity matrix: {csv_path}")
        print(f"  ✓ Saved {filename}")
    
    def _save_cross_era_csv(
        self,
        cross_sim: np.ndarray,
        era1_ids: list,
        era2_ids: list,
        era1_kw: dict,
        era2_kw: dict
    ):
        """Save cross-era topic similarity as sorted CSV."""
        rows = []
        for i, e1_tid in enumerate(era1_ids):
            for j, e2_tid in enumerate(era2_ids):
                rows.append({
                    'era1_topic': e1_tid,
                    'era1_keywords': ', '.join(era1_kw.get(e1_tid, [])[:5]),
                    'era2_topic': e2_tid,
                    'era2_keywords': ', '.join(era2_kw.get(e2_tid, [])[:5]),
                    'similarity': cross_sim[i, j]
                })
        
        df = pd.DataFrame(rows).sort_values('similarity', ascending=False)
        csv_path = self.output_dir / 'topic_similarity.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")
    
    def _print_top_matches(
        self,
        cross_sim: np.ndarray,
        era1_ids: list,
        era2_ids: list,
        era1_kw: dict,
        era2_kw: dict,
        top_n: int = 3
    ):
        """Print top cross-era topic matches."""
        print("\n" + "="*70)
        print("TOP CROSS-ERA TOPIC MATCHES")
        print("="*70)
        
        for i, e1_tid in enumerate(era1_ids):
            top_idx = np.argsort(cross_sim[i])[::-1][:top_n]
            e1_words = ', '.join(era1_kw.get(e1_tid, [])[:5])
            print(f"\nEra1 Topic {e1_tid}: {e1_words}")
            
            for rank, j in enumerate(top_idx, 1):
                e2_tid = era2_ids[j]
                e2_words = ', '.join(era2_kw.get(e2_tid, [])[:5])
                print(f"  {rank}. Era2 T{e2_tid} ({cross_sim[i,j]:.3f}): {e2_words}")
        
        print("="*70)

