"""
Visualization utilities for sentiment analysis results
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def plot_temporal_sentiment_by_topic(
    temporal_sentiments: pd.DataFrame,
    topic_keywords: pd.DataFrame,
    output_path: Optional[Path] = None,
    top_n_topics: Optional[int] = None,
    include_outliers: bool = False
) -> go.Figure:
    """
    Create line plot showing sentiment evolution over time for each topic
    
    Args:
        temporal_sentiments: DataFrame from aggregate_by_topic_temporal()
        topic_keywords: DataFrame with topic keywords for labeling
        output_path: Path to save HTML file (optional)
        top_n_topics: Show only top N topics by document count (optional)
        include_outliers: Include topic -1 (outliers) in plot
    
    Returns:
        Plotly figure object
    """
    df = temporal_sentiments.copy()
    
    # Filter outliers if requested
    if not include_outliers:
        df = df[df['topic'] != -1]
    
    # Select top N topics if specified
    if top_n_topics:
        topic_doc_counts = df.groupby('topic')['doc_count'].sum().sort_values(ascending=False)
        top_topics = topic_doc_counts.head(top_n_topics).index
        df = df[df['topic'].isin(top_topics)]
    
    # Create topic labels with keywords
    topic_labels = {}
    for _, row in topic_keywords.iterrows():
        topic_id = row['Topic']
        keywords = eval(row['Keywords']) if isinstance(row['Keywords'], str) else row['Keywords']
        top_3_keywords = ', '.join(keywords[:3])
        topic_labels[topic_id] = f"Topic {topic_id}: {top_3_keywords}"
    
    # Create figure
    fig = go.Figure()
    
    # Add line for each topic
    for topic_id in sorted(df['topic'].unique()):
        topic_data = df[df['topic'] == topic_id].sort_values('time_period')
        
        label = topic_labels.get(topic_id, f"Topic {topic_id}")
        
        fig.add_trace(go.Scatter(
            x=topic_data['time_period'],
            y=topic_data['mean_sentiment'],
            mode='lines+markers',
            name=label,
            hovertemplate=(
                f'<b>{label}</b><br>' +
                'Period: %{x}<br>' +
                'Sentiment: %{y:.3f}<br>' +
                'Docs: %{customdata[0]}<br>' +
                '<extra></extra>'
            ),
            customdata=topic_data[['doc_count']].values
        ))
    
    # Update layout
    fig.update_layout(
        title='Topic Sentiment Evolution Over Time',
        xaxis_title='Time Period',
        yaxis_title='Mean Sentiment Score',
        yaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
        hovermode='closest',
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05
        )
    )
    
    # Save if path provided
    if output_path:
        fig.write_html(str(output_path))
        logger.info(f"Saved temporal sentiment plot to {output_path}")
    
    return fig


def plot_sentiment_heatmap(
    temporal_sentiments: pd.DataFrame,
    topic_keywords: pd.DataFrame,
    output_path: Optional[Path] = None,
    top_n_topics: Optional[int] = 10
) -> go.Figure:
    """
    Create heatmap showing sentiment across topics and time periods
    
    Args:
        temporal_sentiments: DataFrame from aggregate_by_topic_temporal()
        topic_keywords: DataFrame with topic keywords for labeling
        output_path: Path to save HTML file (optional)
        top_n_topics: Number of topics to show (default: 10)
    
    Returns:
        Plotly figure object
    """
    df = temporal_sentiments.copy()
    
    # Filter outliers
    df = df[df['topic'] != -1]
    
    # Select top N topics by total doc count
    topic_doc_counts = df.groupby('topic')['doc_count'].sum().sort_values(ascending=False)
    top_topics = topic_doc_counts.head(top_n_topics).index
    df = df[df['topic'].isin(top_topics)]
    
    # Create topic labels
    topic_labels = {}
    for _, row in topic_keywords.iterrows():
        topic_id = row['Topic']
        if topic_id in top_topics:
            keywords = eval(row['Keywords']) if isinstance(row['Keywords'], str) else row['Keywords']
            top_3_keywords = ', '.join(keywords[:3])
            topic_labels[topic_id] = f"T{topic_id}: {top_3_keywords}"
    
    # Pivot for heatmap
    heatmap_data = df.pivot(
        index='topic',
        columns='time_period',
        values='mean_sentiment'
    )
    
    # Sort by topic ID
    heatmap_data = heatmap_data.sort_index()
    
    # Create row labels with keywords
    row_labels = [topic_labels.get(topic_id, f"Topic {topic_id}") for topic_id in heatmap_data.index]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=row_labels,
        colorscale='RdYlGn',
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Sentiment'),
        hovertemplate='Topic: %{y}<br>Period: %{x}<br>Sentiment: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Sentiment Heatmap: Top {top_n_topics} Topics Over Time',
        xaxis_title='Time Period',
        yaxis_title='Topic',
        height=max(400, top_n_topics * 40),
        yaxis=dict(tickmode='linear')
    )
    
    if output_path:
        fig.write_html(str(output_path))
        logger.info(f"Saved sentiment heatmap to {output_path}")
    
    return fig


def plot_sentiment_distribution_by_topic(
    topic_sentiments: pd.DataFrame,
    topic_keywords: pd.DataFrame,
    output_path: Optional[Path] = None,
    top_n_topics: Optional[int] = 10
) -> go.Figure:
    """
    Create stacked bar chart showing sentiment distribution for each topic
    
    Args:
        topic_sentiments: DataFrame from aggregate_by_topic()
        topic_keywords: DataFrame with topic keywords
        output_path: Path to save HTML file (optional)
        top_n_topics: Number of topics to show
    
    Returns:
        Plotly figure object
    """
    df = topic_sentiments.copy()
    
    # Filter outliers
    df = df[df['topic'] != -1]
    
    # Select top N topics
    df = df.nlargest(top_n_topics, 'doc_count')
    
    # Create topic labels
    topic_labels = {}
    for _, row in topic_keywords.iterrows():
        topic_id = row['Topic']
        keywords = eval(row['Keywords']) if isinstance(row['Keywords'], str) else row['Keywords']
        top_3_keywords = ', '.join(keywords[:3])
        topic_labels[topic_id] = f"T{topic_id}: {top_3_keywords}"
    
    df['topic_label'] = df['topic'].map(topic_labels)
    df = df.sort_values('mean_sentiment', ascending=False)
    
    # Create stacked bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Positive',
        y=df['topic_label'],
        x=df['positive_pct'] * 100,
        orientation='h',
        marker=dict(color='green'),
        hovertemplate='%{x:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Neutral',
        y=df['topic_label'],
        x=df['neutral_pct'] * 100,
        orientation='h',
        marker=dict(color='gray'),
        hovertemplate='%{x:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Negative',
        y=df['topic_label'],
        x=df['negative_pct'] * 100,
        orientation='h',
        marker=dict(color='red'),
        hovertemplate='%{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Sentiment Distribution by Topic (Top {top_n_topics})',
        xaxis_title='Percentage',
        yaxis_title='Topic',
        barmode='stack',
        height=max(400, top_n_topics * 40),
        showlegend=True
    )
    
    if output_path:
        fig.write_html(str(output_path))
        logger.info(f"Saved sentiment distribution plot to {output_path}")
    
    return fig


def create_all_visualizations(
    era: int,
    output_dir: Path,
    top_n_topics: int = 10
):
    """
    Generate all sentiment visualizations for an era
    
    Args:
        era: Era number (1 or 2)
        output_dir: Output directory (e.g., final_pipeline/outputs/era1)
        top_n_topics: Number of topics to show in plots
    """
    logger.info(f"Creating sentiment visualizations for Era {era}...")
    
    # Load data
    topic_sentiments = pd.read_csv(output_dir / f'Era{era}_topic_sentiments.csv')
    temporal_sentiments = pd.read_csv(output_dir / f'Era{era}_topic_sentiments_temporal.csv')
    topic_keywords = pd.read_csv(output_dir / f'Era{era}_topic_keywords.csv')
    
    # Create visualizations
    logger.info("Creating temporal sentiment plot...")
    plot_temporal_sentiment_by_topic(
        temporal_sentiments,
        topic_keywords,
        output_path=output_dir / f'Era{era}_sentiment_temporal.html',
        top_n_topics=top_n_topics
    )
    
    logger.info("Creating sentiment heatmap...")
    plot_sentiment_heatmap(
        temporal_sentiments,
        topic_keywords,
        output_path=output_dir / f'Era{era}_sentiment_heatmap.html',
        top_n_topics=top_n_topics
    )
    
    logger.info("Creating sentiment distribution plot...")
    plot_sentiment_distribution_by_topic(
        topic_sentiments,
        topic_keywords,
        output_path=output_dir / f'Era{era}_sentiment_distribution.html',
        top_n_topics=top_n_topics
    )
    
    logger.info(f"All visualizations created successfully for Era {era}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sentiment visualizations')
    parser.add_argument('--era', type=int, required=True, choices=[1, 2])
    parser.add_argument('--output-dir', type=Path, default=Path('../outputs'))
    parser.add_argument('--top-n-topics', type=int, default=10)
    
    args = parser.parse_args()
    
    era_output_dir = args.output_dir / f'era{args.era}'
    create_all_visualizations(args.era, era_output_dir, args.top_n_topics)
