"""
Visualization utilities for parameter tuning results
Provides UMAP plots, metric comparisons, and parameter sensitivity analysis
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from umap import UMAP
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ParameterVisualizer:
    """Create visualizations for parameter tuning experiments"""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations (optional)
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_umap_2d(
        self,
        embeddings: np.ndarray,
        topics: np.ndarray,
        documents: List[str] = None,
        metadata: List[Dict] = None,
        title: str = "Topic Distribution (UMAP 2D)",
        show_legend: bool = True,
        save_name: str = None
    ) -> go.Figure:
        """
        Create 2D UMAP scatter plot of documents colored by topic
        
        Args:
            embeddings: Document embeddings
            topics: Topic assignments
            documents: Document texts for hover (optional)
            metadata: Document metadata for hover (optional)
            title: Plot title
            show_legend: Whether to show legend
            save_name: Filename to save plot (optional)
            
        Returns:
            Plotly figure
        """
        logger.info("Generating 2D UMAP projection...")
        
        # Reduce to 2D
        umap_2d = UMAP(
            n_neighbors=15,
            n_components=2,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        coords_2d = umap_2d.fit_transform(embeddings)
        
        # Prepare data
        df = pd.DataFrame({
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'topic': topics,
            'topic_label': [f'Topic {t}' if t != -1 else 'Outlier' for t in topics]
        })
        
        # Add hover information
        if documents:
            df['preview'] = [doc[:200] + '...' if len(doc) > 200 else doc for doc in documents]
        
        if metadata:
            df['source'] = [m.get('source_file', 'Unknown') for m in metadata]
            df['era'] = [m.get('era', 'Unknown') for m in metadata]
        
        # Create plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='topic_label',
            hover_data=['preview', 'source', 'era'] if metadata and documents else None,
            title=title,
            labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        # Style
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(
            width=1000,
            height=700,
            showlegend=show_legend,
            hovermode='closest'
        )
        
        # Save if requested
        if save_name and self.output_dir:
            save_path = self.output_dir / save_name
            fig.write_html(str(save_path))
            logger.info(f"Saved UMAP plot to {save_path}")
        
        return fig
    
    def plot_umap_3d(
        self,
        embeddings: np.ndarray,
        topics: np.ndarray,
        documents: List[str] = None,
        metadata: List[Dict] = None,
        title: str = "Topic Distribution (UMAP 3D)",
        save_name: str = None
    ) -> go.Figure:
        """
        Create 3D UMAP scatter plot of documents colored by topic
        
        Args:
            embeddings: Document embeddings
            topics: Topic assignments
            documents: Document texts for hover (optional)
            metadata: Document metadata for hover (optional)
            title: Plot title
            save_name: Filename to save plot (optional)
            
        Returns:
            Plotly figure
        """
        logger.info("Generating 3D UMAP projection...")
        
        # Reduce to 3D
        umap_3d = UMAP(
            n_neighbors=15,
            n_components=3,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        coords_3d = umap_3d.fit_transform(embeddings)
        
        # Prepare data
        df = pd.DataFrame({
            'x': coords_3d[:, 0],
            'y': coords_3d[:, 1],
            'z': coords_3d[:, 2],
            'topic': topics,
            'topic_label': [f'Topic {t}' if t != -1 else 'Outlier' for t in topics]
        })
        
        # Add hover information
        if documents:
            df['preview'] = [doc[:150] + '...' if len(doc) > 150 else doc for doc in documents]
        
        if metadata:
            df['source'] = [m.get('source_file', 'Unknown') for m in metadata]
        
        # Create plot
        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            color='topic_label',
            hover_data=['preview', 'source'] if metadata and documents else None,
            title=title,
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        # Style
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(
            width=1000,
            height=700,
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3'
            )
        )
        
        # Save if requested
        if save_name and self.output_dir:
            save_path = self.output_dir / save_name
            fig.write_html(str(save_path))
            logger.info(f"Saved 3D UMAP plot to {save_path}")
        
        return fig
    
    def plot_metric_comparison(
        self,
        results_df: pd.DataFrame,
        parameter_name: str,
        metrics: List[str] = None,
        title: str = None,
        save_name: str = None
    ) -> go.Figure:
        """
        Plot multiple metrics across parameter values
        
        Args:
            results_df: DataFrame with tuning results
            parameter_name: Parameter to plot on x-axis
            metrics: List of metrics to plot (optional, defaults to main metrics)
            title: Plot title (optional)
            save_name: Filename to save plot (optional)
            
        Returns:
            Plotly figure
        """
        if metrics is None:
            metrics = ['coherence', 'diversity', 'silhouette', 'composite_score']
        
        # Create subplots
        n_metrics = len(metrics)
        fig = make_subplots(
            rows=(n_metrics + 1) // 2,
            cols=2,
            subplot_titles=[m.replace('_', ' ').title() for m in metrics],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Sort by parameter value
        df = results_df.sort_values(parameter_name)
        
        # Add traces
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=df[parameter_name],
                    y=df[metric],
                    mode='lines+markers',
                    name=metric,
                    marker=dict(size=10),
                    line=dict(width=2),
                    showlegend=False
                ),
                row=row,
                col=col
            )
            
            # Add axis labels
            fig.update_xaxes(title_text=parameter_name, row=row, col=col)
            fig.update_yaxes(title_text=metric, row=row, col=col)
        
        # Update layout
        plot_title = title or f"Metrics vs {parameter_name}"
        fig.update_layout(
            title_text=plot_title,
            height=400 * ((n_metrics + 1) // 2),
            width=1000,
            showlegend=False
        )
        
        # Save if requested
        if save_name and self.output_dir:
            save_path = self.output_dir / save_name
            fig.write_html(str(save_path))
            logger.info(f"Saved metric comparison to {save_path}")
        
        return fig
    
    def plot_parameter_heatmap(
        self,
        results_df: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str = 'composite_score',
        title: str = None,
        save_name: str = None
    ) -> go.Figure:
        """
        Create heatmap showing metric values across two parameters
        
        Args:
            results_df: DataFrame with tuning results
            param1: First parameter for x-axis
            param2: Second parameter for y-axis
            metric: Metric to visualize
            title: Plot title (optional)
            save_name: Filename to save plot (optional)
            
        Returns:
            Plotly figure
        """
        # Pivot data
        pivot_df = results_df.pivot_table(
            values=metric,
            index=param2,
            columns=param1,
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Viridis',
            colorbar=dict(title=metric.replace('_', ' ').title()),
            text=np.round(pivot_df.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        # Update layout
        plot_title = title or f"{metric.replace('_', ' ').title()} Heatmap"
        fig.update_layout(
            title=plot_title,
            xaxis_title=param1,
            yaxis_title=param2,
            width=800,
            height=600
        )
        
        # Save if requested
        if save_name and self.output_dir:
            save_path = self.output_dir / save_name
            fig.write_html(str(save_path))
            logger.info(f"Saved heatmap to {save_path}")
        
        return fig
    
    def plot_topic_distribution(
        self,
        topics: np.ndarray,
        title: str = "Topic Distribution",
        save_name: str = None
    ) -> go.Figure:
        """
        Plot histogram of documents per topic
        
        Args:
            topics: Topic assignments
            title: Plot title
            save_name: Filename to save plot (optional)
            
        Returns:
            Plotly figure
        """
        # Count documents per topic
        topic_counts = pd.Series(topics).value_counts().sort_index()
        
        # Separate outliers
        outlier_count = topic_counts.get(-1, 0)
        topic_counts = topic_counts[topic_counts.index != -1]
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f'Topic {t}' for t in topic_counts.index],
            y=topic_counts.values,
            name='Topics',
            marker_color='steelblue'
        ))
        
        if outlier_count > 0:
            fig.add_trace(go.Bar(
                x=['Outliers'],
                y=[outlier_count],
                name='Outliers',
                marker_color='coral'
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Topic',
            yaxis_title='Number of Documents',
            width=1000,
            height=500,
            showlegend=True
        )
        
        # Save if requested
        if save_name and self.output_dir:
            save_path = self.output_dir / save_name
            fig.write_html(str(save_path))
            logger.info(f"Saved topic distribution to {save_path}")
        
        return fig
    
    def plot_configuration_comparison(
        self,
        results_df: pd.DataFrame,
        top_n: int = 10,
        metrics: List[str] = None,
        save_name: str = None
    ) -> go.Figure:
        """
        Compare top N configurations across multiple metrics
        
        Args:
            results_df: DataFrame with tuning results
            top_n: Number of top configurations to show
            metrics: Metrics to compare (optional)
            save_name: Filename to save plot (optional)
            
        Returns:
            Plotly figure
        """
        if metrics is None:
            metrics = ['coherence', 'diversity', 'n_topics', 'outlier_pct', 'silhouette']
        
        # Get top N by composite score
        top_configs = results_df.nlargest(top_n, 'composite_score')
        
        # Normalize metrics to 0-1 scale for comparison
        normalized_data = {}
        for metric in metrics:
            values = top_configs[metric].values
            if metric == 'outlier_pct':
                # Lower is better for outliers
                normalized = 1 - (values / values.max() if values.max() > 0 else values)
            elif metric == 'n_topics':
                # Closer to 20 is better
                target = 20
                normalized = 1 / (1 + np.abs(values - target) / target)
            else:
                # Higher is better
                min_val, max_val = values.min(), values.max()
                normalized = (values - min_val) / (max_val - min_val) if max_val > min_val else values
            
            normalized_data[metric] = normalized
        
        # Create radar chart
        fig = go.Figure()
        
        for idx, row_idx in enumerate(top_configs.index[:top_n]):
            values = [normalized_data[m][idx] for m in metrics]
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                name=f'Config {idx + 1}',
                fill='toself',
                opacity=0.6
            ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f"Top {top_n} Configuration Comparison (Normalized Metrics)",
            width=900,
            height=700,
            showlegend=True
        )
        
        # Save if requested
        if save_name and self.output_dir:
            save_path = self.output_dir / save_name
            fig.write_html(str(save_path))
            logger.info(f"Saved configuration comparison to {save_path}")
        
        return fig
    
    def create_summary_dashboard(
        self,
        results_df: pd.DataFrame,
        embeddings: np.ndarray,
        topics: np.ndarray,
        documents: List[str] = None,
        metadata: List[Dict] = None,
        save_name: str = "tuning_dashboard.html"
    ):
        """
        Create comprehensive dashboard with multiple visualizations
        
        Args:
            results_df: DataFrame with tuning results
            embeddings: Document embeddings
            topics: Topic assignments for best config
            documents: Document texts (optional)
            metadata: Document metadata (optional)
            save_name: Filename to save dashboard
        """
        logger.info("Creating comprehensive tuning dashboard...")
        
        # Create individual plots
        umap_fig = self.plot_umap_2d(
            embeddings, 
            topics, 
            documents, 
            metadata,
            title="Best Configuration - Topic Distribution"
        )
        
        dist_fig = self.plot_topic_distribution(
            topics,
            title="Best Configuration - Documents per Topic"
        )
        
        comparison_fig = self.plot_configuration_comparison(
            results_df,
            top_n=5
        )
        
        # Combine into dashboard (simple HTML concatenation)
        if self.output_dir:
            dashboard_path = self.output_dir / save_name
            
            with open(dashboard_path, 'w') as f:
                f.write("<html><head><title>Parameter Tuning Dashboard</title></head><body>")
                f.write("<h1 style='text-align: center;'>BERTopic Parameter Tuning Dashboard</h1>")
                
                # Best configuration summary
                best_row = results_df.nlargest(1, 'composite_score').iloc[0]
                f.write("<div style='margin: 20px; padding: 20px; background-color: #f0f0f0;'>")
                f.write("<h2>Best Configuration</h2>")
                f.write(f"<p><b>Composite Score:</b> {best_row['composite_score']:.3f}</p>")
                f.write(f"<p><b>Coherence:</b> {best_row['coherence']:.3f}</p>")
                f.write(f"<p><b>Diversity:</b> {best_row['diversity']:.3f}</p>")
                f.write(f"<p><b>Number of Topics:</b> {best_row['n_topics']}</p>")
                f.write(f"<p><b>Outlier Percentage:</b> {best_row['outlier_pct']:.1%}</p>")
                f.write(f"<p><b>Silhouette Score:</b> {best_row['silhouette']:.3f}</p>")
                f.write("</div>")
                
                # Add plots
                f.write(umap_fig.to_html(full_html=False, include_plotlyjs='cdn'))
                f.write(dist_fig.to_html(full_html=False, include_plotlyjs=False))
                f.write(comparison_fig.to_html(full_html=False, include_plotlyjs=False))
                
                f.write("</body></html>")
            
            logger.info(f"Saved dashboard to {dashboard_path}")
