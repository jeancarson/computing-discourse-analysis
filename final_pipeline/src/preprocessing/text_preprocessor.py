import json
import logging
from pathlib import Path
from typing import List, Dict

try:
    from .config import load_preprocessing_config, PreprocessingConfig
    from .data_loader import DataLoader
    from .text_utils import (
        dehyphenate_text,
        normalize_whitespace,
        lemmatize_text,
        remove_stopwords,
        remove_special_characters
    )
except ImportError:
    from config import load_preprocessing_config, PreprocessingConfig
    from data_loader import DataLoader
    from text_utils import (
        dehyphenate_text,
        normalize_whitespace,
        lemmatize_text,
        remove_stopwords,
        remove_special_characters
    )

logger = logging.getLogger(__name__)


class TextPreprocessor:

    def __init__(self, config: PreprocessingConfig | dict | None = None):
        if config is None:
            self.config = load_preprocessing_config()
        elif isinstance(config, PreprocessingConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = PreprocessingConfig(**config)
        else:
            self.config = load_preprocessing_config()

        stopwords_file = Path(__file__).parent / "stopwords.json"
        self.custom_stopwords = []
        if stopwords_file.exists():
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                self.custom_stopwords = json.load(f)

    def preprocess_text(self, text: str) -> str:
        cfg = self.config

        if getattr(cfg, 'dehyphenate', True):
            text = dehyphenate_text(text)

        if getattr(cfg, 'normalize_whitespace', True):
            text = normalize_whitespace(text)

        if getattr(cfg, 'lemmatize', True):
            text = lemmatize_text(text)

        if getattr(cfg, 'remove_special_chars', True):
            text = remove_special_characters(text)

        if getattr(cfg, 'remove_stopwords', True):
            text = remove_stopwords(text, self.custom_stopwords)

        if getattr(cfg, 'lowercase', True):
            text = text.lower()

        return text.strip()

    def save_processed_texts(self, processed_texts: List[str], metadata_list: List[dict], output_dir: str, era: str):
        processed_dir = Path(output_dir) / "processed" / era
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        text_files = []
        for i, (text, metadata) in enumerate(zip(processed_texts, metadata_list)):
            # Flatten the full relative path into a unique filename to avoid
            # collisions (e.g. multiple "section_01.txt" from different CACM/Datamation issues)
            original_filepath = metadata.get('filepath', '')
            if original_filepath:
                # e.g. "Era1/CACM/cacm_1958_08/section_01.txt" -> "cacm_1958_08_section_01.txt"
                # Strip the era prefix (already implied by the output directory) and flatten
                parts = Path(original_filepath).parts
                # Drop the era folder and source-type folder, keep issue + filename
                # e.g. ('Era1', 'CACM', 'cacm_1958_08', 'section_01.txt') -> 'cacm_1958_08_section_01.txt'
                if len(parts) > 2:
                    text_filename = '_'.join(parts[2:])  # skip era + source type
                else:
                    text_filename = '_'.join(parts)
                # Ensure .txt extension
                if not text_filename.endswith('.txt'):
                    text_filename += '.txt'
            else:
                # Fallback to generic naming if filepath is missing
                text_filename = f"{era}_doc_{i:04d}.txt"
            
            text_path = processed_dir / text_filename
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            text_files.append({
                "index": i,
                "text_file": text_filename,
                "original_filepath": original_filepath,  # Keep reference to cleaned source
                "metadata": metadata
            })
        
        index_file = processed_dir / f"{era}_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(text_files, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(processed_texts)} processed documents to {processed_dir}")

    def load_processed_texts(self, output_dir: str, era: str) -> tuple[List[str], List[dict]]:
        processed_dir = Path(output_dir) / "processed" / era
        index_file = processed_dir / f"{era}_index.json"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Processed data not found at {index_file}")
        
        with open(index_file, 'r', encoding='utf-8') as f:
            text_files = json.load(f)
        
        processed_texts = []
        metadata_list = []
        
        for item in text_files:
            text_path = processed_dir / item["text_file"]
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            processed_texts.append(text)
            metadata_list.append(item["metadata"])
        
        logger.info(f"Loaded {len(processed_texts)} processed documents from {processed_dir}")
        return processed_texts, metadata_list


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    repo_root = Path(__file__).resolve().parents[3]
    cleaned_root = repo_root / 'data' / 'cleaned'
    output_root = repo_root / 'data'

    if not cleaned_root.exists():
        logger.error(f"Cleaned data directory not found: {cleaned_root}")
        return

    cfg = load_preprocessing_config()
    preprocessor = TextPreprocessor(cfg)
    data_loader = DataLoader(cleaned_root)
    min_len = cfg.min_doc_length

    for era in ['Era1', 'Era2']:
        logger.info(f"Processing {era}...")
        
        try:
            texts, metadata_list = data_loader.load_era_data(era)
        except Exception as e:
            logger.error(f"Failed to load data for {era}: {e}")
            continue

        if not texts:
            logger.info(f"No documents found for {era}")
            continue

        processed_texts = []
        processed_metadata = []
        skipped = 0

        for text, metadata in zip(texts, metadata_list):
            try:
                processed = preprocessor.preprocess_text(text)
                
                if len(processed.split()) < min_len:
                    skipped += 1
                    continue
                
                processed_texts.append(processed)
                processed_metadata.append(metadata)
                
            except Exception as e:
                logger.warning(f"Preprocessing failed for {metadata.get('filepath', 'unknown')}: {e}")
                skipped += 1

        if processed_texts:
            preprocessor.save_processed_texts(processed_texts, processed_metadata, str(output_root), era)
            logger.info(f"Completed {era}: {len(processed_texts)} docs saved, {skipped} skipped")
        else:
            logger.info(f"No documents passed preprocessing for {era}")


if __name__ == '__main__':
    main()