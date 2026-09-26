"""
BERTopic Configuration with Pydantic validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import json
from pathlib import Path


class UMAPConfig(BaseModel):
    n_neighbors: int = Field(5, description="Number of neighbors for UMAP")
    n_components: int = Field(3, description="Number of components for UMAP")
    min_dist: float = Field(0.05, description="Minimum distance for UMAP")
    random_state: int = Field(42, description="Random state for reproducibility")


class HDBSCANConfig(BaseModel):
    min_cluster_size: int = Field(15, description="Minimum cluster size for HDBSCAN")
    min_samples: int = Field(10, description="Minimum samples for HDBSCAN")
    gen_min_span_tree: bool = Field(True, description="Generate minimum spanning tree")
    prediction_data: bool = Field(True, description="Generate prediction data")


class VectorizerConfig(BaseModel):
    ngram_range: Tuple[int, int] = Field((1, 2), description="N-gram range for vectorizer")
    stop_words: str = Field("english", description="Stop words language")
    additional_stopwords: List[str] = Field(
        ["http", "https", "amp", "com", "www"],
        description="Additional stopwords to remove"
    )


class BERTopicConfig(BaseModel):
    top_n_words: int = Field(10, description="Number of top words per topic")
    language: str = Field("english", description="Language for BERTopic")
    calculate_probabilities: bool = Field(True, description="Calculate topic probabilities")
    verbose: bool = Field(True, description="Verbose output")


class PreprocessingConfig(BaseModel):
    remove_headers: bool = Field(True, description="Remove text headers")
    normalize_whitespace: bool = Field(True, description="Normalize whitespace")
    remove_stopwords: bool = Field(True, description="Remove stopwords")
    lemmatize: bool = Field(True, description="Lemmatize text")
    min_text_length: Optional[int] = Field(None, description="Minimum text length (None = no filter)")


class MultiTopicAssignmentConfig(BaseModel):
    enabled: bool = Field(False, description="Enable multi-topic assignment")
    probability_threshold: float = Field(0.2, description="Minimum probability for topic assignment")


class TopicmodellingConfig(BaseModel):
    embedding_model: str = Field("all-MiniLM-L6-v2", description="Sentence transformer model")
    umap: UMAPConfig = Field(default_factory=UMAPConfig)
    hdbscan: HDBSCANConfig = Field(default_factory=HDBSCANConfig)
    vectorizer: VectorizerConfig = Field(default_factory=VectorizerConfig)
    bertopic: BERTopicConfig = Field(default_factory=BERTopicConfig)
    multi_topic_assignment: MultiTopicAssignmentConfig = Field(default_factory=MultiTopicAssignmentConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)


def load_config(config_path: Path | str = None, era: int = None) -> TopicmodellingConfig:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file. If None, uses default src/config.json
        era: Era number (1 or 2). If provided, merges shared config with era-specific config
    
    Returns:
        TopicmodellingConfig instance
    """
    if config_path is None:
        # Default to src/config.json
        config_path = Path(__file__).parent / "config.json"
    
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Handle centralized config with era-specific sections
    if "shared" in config_dict and era is not None:
        if f"era{era}" not in config_dict:
            raise ValueError(f"Era {era} not found in config. Available eras: {[k for k in config_dict.keys() if k.startswith('era')]}")
        
        # Merge shared config with era-specific config
        merged_config = {**config_dict["shared"]}
        merged_config.update(config_dict[f"era{era}"])
        config_dict = merged_config
    elif "shared" in config_dict:
        # If no era specified but config has shared section, use shared only
        config_dict = config_dict["shared"]

    return TopicmodellingConfig(**config_dict)


def save_config(config: TopicmodellingConfig, config_path: Path):
    """Save configuration to JSON file"""
    with open(config_path, 'w') as f:
        json.dump(config.model_dump(), f, indent=2)


# Default configuration
DEFAULT_CONFIG = TopicmodellingConfig()
