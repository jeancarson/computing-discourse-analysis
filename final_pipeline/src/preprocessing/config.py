from pydantic import BaseModel, Field
from pathlib import Path
import json

class PreprocessingConfig(BaseModel):
    dehyphenate: bool = Field(True, description="De-hyphenate split words")
    lemmatize: bool = Field(True, description="Apply lemmatization")
    normalize_whitespace: bool = Field(True, description="Normalize whitespace")
    lowercase: bool = Field(True, description="Convert text to lowercase")
    remove_special_chars: bool = Field(True, description="Remove special characters and digits")
    remove_stopwords: bool = Field(True, description="Remove stopwords")
    min_doc_length: int = Field(50, description="Minimum document word count to keep")
    


def load_preprocessing_config(config_path: Path | str = None) -> PreprocessingConfig:
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return PreprocessingConfig()

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    return PreprocessingConfig(**cfg)
