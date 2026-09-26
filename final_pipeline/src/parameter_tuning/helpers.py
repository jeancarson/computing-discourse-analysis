"""
Parameter tuning helpers for BERTopic optimization
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
from itertools import product
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from topic_modelling.config import TopicmodellingConfig, load_config, save_config
from topic_modelling.core.topic_modeler import TopicModeler

logger = logging.getLogger(__name__)


class ParameterTuner:
    """Manage parameter tuning experiments for BERTopic"""
    
    def __init__(self, base_config: TopicmodellingConfig):
        """
        Initialize parameter tuner
        
        Args:
            base_config: Base configuration to start from
        """
        self.base_config = base_config
        self.results = []
        
    def evaluate_model(
        self, 
        topic_model: BERTopic,
        documents: List[str],
        embeddings: np.ndarray,
        topics: np.ndarray,
        probs: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Evaluate topic model with multiple metrics
        
        Args:
            topic_model: Fitted BERTopic model
            documents: List of document texts
            embeddings: Document embeddings
            topics: Topic assignments
            probs: Topic probabilities (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure topics and embeddings are numpy arrays
        topics = np.array(topics)
        embeddings = np.array(embeddings)
        if probs is not None:
            probs = np.array(probs)
        
        metrics = {}
        
        # 1. Topic Coherence (C_V metric from Paper 2)
        try:
            coherence = self._calculate_coherence(topic_model, documents, topics)
            metrics['coherence'] = coherence
        except Exception as e:
            logger.warning(f"Coherence calculation failed: {e}")
            metrics['coherence'] = 0.0
        
        # 2. Topic Diversity (Paper 2)
        try:
            diversity = self._calculate_diversity(topic_model, topics)
            metrics['diversity'] = diversity
        except Exception as e:
            logger.warning(f"Diversity calculation failed: {e}")
            metrics['diversity'] = 0.0
        
        # 3. Number of topics (excluding outlier -1)
        unique_topics = set(topics)
        n_topics = len(unique_topics) - (1 if -1 in unique_topics else 0)
        metrics['n_topics'] = n_topics
        
        # 4. Outlier percentage
        outlier_pct = (topics == -1).sum() / len(topics)
        metrics['outlier_pct'] = outlier_pct
        
        # 5. Average documents per topic
        if n_topics > 0:
            non_outlier_docs = (topics != -1).sum()
            metrics['avg_docs_per_topic'] = non_outlier_docs / n_topics
        else:
            metrics['avg_docs_per_topic'] = 0
        
        # 6. Silhouette score (cluster separation)
        try:
            if len(unique_topics) > 1:
                # Only use non-outlier documents for silhouette
                mask = topics != -1
                if mask.sum() > 1:
                    silhouette = silhouette_score(
                        embeddings[mask], 
                        topics[mask],
                        metric='euclidean'
                    )
                    metrics['silhouette'] = silhouette
                else:
                    metrics['silhouette'] = 0.0
            else:
                metrics['silhouette'] = 0.0
        except Exception as e:
            logger.warning(f"Silhouette score calculation failed: {e}")
            metrics['silhouette'] = 0.0
        
        # 7. Topic size statistics
        topic_counts = pd.Series(topics[topics != -1]).value_counts()
        metrics['min_topic_size'] = int(topic_counts.min()) if len(topic_counts) > 0 else 0
        metrics['max_topic_size'] = int(topic_counts.max()) if len(topic_counts) > 0 else 0
        metrics['median_topic_size'] = int(topic_counts.median()) if len(topic_counts) > 0 else 0
        
        return metrics
    
    def _calculate_coherence(
        self, 
        topic_model: BERTopic, 
        documents: List[str], 
        topics: np.ndarray
    ) -> float:
        """
        Calculate topic coherence using Gensim's C_V metric
        
        Args:
            topic_model: Fitted BERTopic model
            documents: List of document texts
            topics: Topic assignments
            
        Returns:
            Coherence score (higher is better, typically 0-1)
        """
        # Tokenize documents
        texts = [doc.lower().split() for doc in documents]
        dictionary = Dictionary(texts)
        
        # Get topic words (excluding outlier topic -1)
        topic_words = []
        for topic_id in set(topics):
            if topic_id != -1:
                words = [w for w, _ in topic_model.get_topic(topic_id)[:10]]
                if words:
                    topic_words.append(words)
        
        if not topic_words:
            return 0.0
        
        # Calculate coherence
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        
        return coherence_model.get_coherence()
    
    def _calculate_diversity(
        self, 
        topic_model: BERTopic, 
        topics: np.ndarray
    ) -> float:
        """
        Calculate topic diversity (unique words / total words)
        
        Args:
            topic_model: Fitted BERTopic model
            topics: Topic assignments
            
        Returns:
            Diversity score (0-1, higher is better)
        """
        unique_words = set()
        total_words = 0
        
        for topic_id in set(topics):
            if topic_id != -1:
                words = [w for w, _ in topic_model.get_topic(topic_id)[:10]]
                unique_words.update(words)
                total_words += len(words)
        
        if total_words == 0:
            return 0.0
        
        return len(unique_words) / total_words
    
    def calculate_composite_score(
        self, 
        metrics: Dict[str, Any],
        weights: Dict[str, float] = None
    ) -> float:
        """
        Calculate weighted composite score for model comparison
        
        Args:
            metrics: Dictionary of evaluation metrics
            weights: Custom weights for each metric (optional)
            
        Returns:
            Composite score (higher is better)
        """
        if weights is None:
            # Default weights from paper recommendations
            weights = {
                'coherence': 0.40,      # Most important
                'diversity': 0.25,      # Important for topic separation
                'outlier_pct': 0.15,    # Prefer fewer outliers
                'silhouette': 0.10,     # Cluster quality
                'n_topics': 0.10        # Target ~20 topics
            }
        
        score = 0.0
        
        # Coherence (0-1, higher is better)
        score += weights['coherence'] * metrics.get('coherence', 0)
        
        # Diversity (0-1, higher is better)
        score += weights['diversity'] * metrics.get('diversity', 0)
        
        # Outlier percentage (0-1, lower is better)
        outlier_penalty = 1 - metrics.get('outlier_pct', 1.0)
        score += weights['outlier_pct'] * outlier_penalty
        
        # Silhouette (typically -1 to 1, normalize to 0-1)
        silhouette_norm = (metrics.get('silhouette', 0) + 1) / 2
        score += weights['silhouette'] * silhouette_norm
        
        # Number of topics (penalize distance from target of 10 for 165 docs)
        n_topics = metrics.get('n_topics', 0)
        target_topics = 10
        if n_topics > 0:
            topic_score = 1 / (1 + abs(n_topics - target_topics) / target_topics)
        else:
            topic_score = 0
        score += weights['n_topics'] * topic_score
        
        return score
    
    def sensitivity_test(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        parameter_name: str,
        parameter_values: List[Any],
        config_updates: Dict[str, List] = None
    ) -> pd.DataFrame:
        """
        Test sensitivity to a single parameter
        
        Args:
            documents: List of document texts
            embeddings: Pre-computed document embeddings
            parameter_name: Name of parameter to test
            parameter_values: List of values to test
            config_updates: Additional config updates for each test (optional)
            
        Returns:
            DataFrame with results for each parameter value
        """
        results = []
        
        import torch
        import gc
        
        for i, value in enumerate(parameter_values):
            # Aggressive memory cleanup BEFORE each test
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"Testing {parameter_name}={value}")
            
            # Create config copy
            config = self.base_config.model_copy(deep=True)
            
            # Update parameter
            self._update_config_parameter(config, parameter_name, value)
            
            # Apply additional updates if provided
            if config_updates:
                for key, values in config_updates.items():
                    self._update_config_parameter(config, key, values[i])
            
            # Create and fit model
            modeler = TopicModeler(config)
            modeler.create_model()
            topics, probs = modeler.fit_transform(documents, embeddings=embeddings)
            
            # Evaluate
            metrics = self.evaluate_model(
                modeler.model,
                documents,
                embeddings,
                topics,
                probs
            )
            
            # Add parameter info
            metrics['parameter'] = parameter_name
            metrics['value'] = value
            metrics['composite_score'] = self.calculate_composite_score(metrics)
            
            results.append(metrics)
            
            # Log results
            logger.info(
                f"  → Topics={metrics['n_topics']}, "
                f"Coherence={metrics['coherence']:.3f}, "
                f"Diversity={metrics['diversity']:.3f}, "
                f"Outliers={metrics['outlier_pct']:.1%}, "
                f"Score={metrics['composite_score']:.3f}"
            )
            
            # Aggressive cleanup after each test
            del modeler, topics, probs, metrics
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        return pd.DataFrame(results)
    
    def grid_search(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        param_grid: Dict[str, List[Any]],
        max_combinations: int = None
    ) -> Tuple[pd.DataFrame, TopicmodellingConfig]:
        """
        Perform grid search over parameter combinations
        
        Args:
            documents: List of document texts
            embeddings: Pre-computed document embeddings
            param_grid: Dictionary mapping parameter names to value lists
            max_combinations: Maximum combinations to test (optional)
            
        Returns:
            Tuple of (results DataFrame, best configuration)
        """
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        if max_combinations and len(combinations) > max_combinations:
            logger.warning(
                f"Grid has {len(combinations)} combinations, "
                f"limiting to {max_combinations}"
            )
            combinations = combinations[:max_combinations]
        
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        results = []
        best_score = -1
        best_config = None
        
        import torch
        import gc
        
        for i, param_combo in enumerate(combinations, 1):
            # Aggressive memory cleanup BEFORE each iteration
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"\n[{i}/{len(combinations)}] Testing combination:")
            
            # Create config copy
            config = self.base_config.model_copy(deep=True)
            
            # Update all parameters
            param_dict = {}
            for param_name, param_value in zip(param_names, param_combo):
                self._update_config_parameter(config, param_name, param_value)
                param_dict[param_name] = param_value
                logger.info(f"  {param_name}={param_value}")
            
            # Create and fit model
            try:
                modeler = TopicModeler(config)
                modeler.create_model()
                topics, probs = modeler.fit_transform(documents, embeddings=embeddings)
                
                # Evaluate
                metrics = self.evaluate_model(
                    modeler.model,
                    documents,
                    embeddings,
                    topics,
                    probs
                )
                
                # Add parameter info
                metrics.update(param_dict)
                metrics['composite_score'] = self.calculate_composite_score(metrics)
                
                results.append(metrics)
                
                # Track best
                if metrics['composite_score'] > best_score:
                    best_score = metrics['composite_score']
                    best_config = config
                
                # Log results
                logger.info(
                    f"  → Topics={metrics['n_topics']}, "
                    f"Coherence={metrics['coherence']:.3f}, "
                    f"Diversity={metrics['diversity']:.3f}, "
                    f"Score={metrics['composite_score']:.3f}"
                )
                
                # Aggressive cleanup after each iteration
                del modeler, topics, probs, metrics
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
            except Exception as e:
                logger.error(f"Failed for combination {param_dict}: {e}")
                # Aggressive cleanup even on failure
                if 'modeler' in locals():
                    del modeler
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                continue
        
        results_df = pd.DataFrame(results)
        
        # Sort by composite score
        results_df = results_df.sort_values('composite_score', ascending=False)
        
        logger.info(f"\nBest configuration (score={best_score:.3f}):")
        for param_name in param_names:
            best_value = self._get_config_parameter(best_config, param_name)
            logger.info(f"  {param_name}={best_value}")
        
        return results_df, best_config
    
    def _update_config_parameter(
        self, 
        config: TopicmodellingConfig, 
        param_path: str, 
        value: Any
    ):
        """
        Update nested config parameter using dot notation
        
        Args:
            config: Configuration object to update
            param_path: Parameter path (e.g., 'umap.n_neighbors')
            value: New value
        """
        parts = param_path.split('.')
        
        if len(parts) == 2:
            # Nested parameter (e.g., umap.n_neighbors)
            section, param = parts
            section_obj = getattr(config, section)
            setattr(section_obj, param, value)
        else:
            # Top-level parameter
            setattr(config, param_path, value)
    
    def _get_config_parameter(
        self, 
        config: TopicmodellingConfig, 
        param_path: str
    ) -> Any:
        """
        Get nested config parameter using dot notation
        
        Args:
            config: Configuration object
            param_path: Parameter path (e.g., 'umap.n_neighbors')
            
        Returns:
            Parameter value
        """
        parts = param_path.split('.')
        
        if len(parts) == 2:
            section, param = parts
            section_obj = getattr(config, section)
            return getattr(section_obj, param)
        else:
            return getattr(config, param_path)
    
    def save_results(
        self, 
        results_df: pd.DataFrame, 
        output_path: Path,
        best_config: TopicmodellingConfig = None,
        experiment_name: str = None
    ):
        """
        Save tuning results to disk
        
        Args:
            results_df: DataFrame with results
            output_path: Directory to save results
            best_config: Best configuration to save (optional)
            experiment_name: Name for this experiment (optional)
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = experiment_name or "tuning"
        
        # Save results CSV
        csv_path = output_path / f"{exp_name}_{timestamp}_results.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to {csv_path}")
        
        # Save best config
        if best_config:
            config_path = output_path / f"{exp_name}_{timestamp}_best_config.json"
            save_config(best_config, config_path)
            logger.info(f"Saved best config to {config_path}")
        
        # Save summary
        summary_path = output_path / f"{exp_name}_{timestamp}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Parameter Tuning Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total configurations tested: {len(results_df)}\n\n")
            
            # Top 5 configurations
            f.write("Top 5 Configurations:\n")
            f.write("-" * 60 + "\n")
            top5 = results_df.nlargest(5, 'composite_score')
            for idx, row in top5.iterrows():
                f.write(f"\nRank {top5.index.get_loc(idx) + 1}:\n")
                f.write(f"  Composite Score: {row['composite_score']:.3f}\n")
                f.write(f"  Coherence: {row['coherence']:.3f}\n")
                f.write(f"  Diversity: {row['diversity']:.3f}\n")
                f.write(f"  N Topics: {row['n_topics']}\n")
                f.write(f"  Outliers: {row['outlier_pct']:.1%}\n")
                f.write(f"  Silhouette: {row['silhouette']:.3f}\n")
        
        logger.info(f"Saved summary to {summary_path}")


def generate_embeddings(
    documents: List[str],
    model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Generate document embeddings with explicit progress updates
    
    Args:
        documents: List of document texts
        model_name: Sentence transformer model name
        
    Returns:
        Document embeddings array
    """
    import sys
    import time
    
    logger.info(f"Generating embeddings with {model_name}")
    embedding_model = SentenceTransformer(model_name, trust_remote_code=True)
    
    # Manual batching with print-based progress (works reliably in all notebooks)
    batch_size = 32
    all_embeddings = []
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    print(f"\n🔄 Processing {len(documents)} documents in {total_batches} batches...")
    start_time = time.time()
    
    for i in range(0, len(documents), batch_size):
        batch_num = (i // batch_size) + 1
        batch = documents[i:i + batch_size]
        
        batch_start = time.time()
        batch_embeddings = embedding_model.encode(
            batch, 
            show_progress_bar=False,
            convert_to_numpy=True
        )
        batch_time = time.time() - batch_start
        
        all_embeddings.append(batch_embeddings)
        
        # Print progress after each batch
        elapsed = time.time() - start_time
        docs_done = min(i + batch_size, len(documents))
        pct = (docs_done / len(documents)) * 100
        
        print(f"  ✓ Batch {batch_num}/{total_batches} ({docs_done}/{len(documents)} docs, {pct:.1f}%) - {batch_time:.1f}s")
        sys.stdout.flush()  # Force output to display immediately
    
    total_time = time.time() - start_time
    print(f"✅ Complete! Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")
    
    return np.vstack(all_embeddings)
