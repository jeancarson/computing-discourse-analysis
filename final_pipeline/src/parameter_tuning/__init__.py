"""
Parameter tuning package for BERTopic optimization

This package provides tools for systematic parameter tuning:
- helpers.py: Evaluation metrics and grid search
- visualizations.py: UMAP plots and metric comparisons
- tuning_notebook.ipynb: Interactive tuning workflow
"""

from .helpers import ParameterTuner, generate_embeddings
from .visualizations import ParameterVisualizer

__all__ = ['ParameterTuner', 'ParameterVisualizer', 'generate_embeddings']
