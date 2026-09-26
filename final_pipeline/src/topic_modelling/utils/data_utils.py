import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def load_processed_data(data_root: Path, era: int) -> Tuple[List[str], List[Dict]]:
    """
    Load preprocessed data from processed directory.
    
    Args:
        data_root: Path to data directory (e.g., '../../data')
        era: Era number (1 or 2)
    
    Returns:
        Tuple of (texts, metadata_list)
    """
    era_str = f'Era{era}'
    processed_dir = data_root / 'processed' / era_str
    index_file = processed_dir / f"{era_str}_index.json"
    
    if not index_file.exists():
        raise FileNotFoundError(f"Processed index not found: {index_file}")
    
    with open(index_file, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    texts = []
    metadata_list = []
    
    for item in index_data:
        text_path = processed_dir / item["text_file"]
        with open(text_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
        metadata_list.append(item["metadata"])
    
    logger.info(f"Loaded {len(texts)} processed documents from Era {era}")
    return texts, metadata_list
