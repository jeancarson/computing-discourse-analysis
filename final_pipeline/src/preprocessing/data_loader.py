"""
Data loading utilities - modular and framework-agnostic
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and organize text data from CleanedSources"""

    def __init__(self, cleaned_sources_path: Path):
        self.cleaned_sources_path = cleaned_sources_path

    def load_era_data(
        self, 
        era: str, 
        year_start: Optional[int] = None, 
        year_end: Optional[int] = None
    ) -> Tuple[List[str], List[Dict]]:
        """
        Load text files from an era's subdirectories, optionally filtered by year range

        Args:
            era: Era identifier (e.g., 'Era1', 'Era2')
            year_start: Start year for filtering (inclusive)
            year_end: End year for filtering (inclusive). If None, only year_start is used.

        Returns:
            Tuple of (texts, metadata_list)
        """
        era_path = self.cleaned_sources_path / era
        if not era_path.exists():
            raise FileNotFoundError(f"Era directory not found: {era_path}")

        texts = []
        metadata_list = []

        # Find all JSON files
        json_files = list(era_path.rglob("*Cleaned.json"))
        logger.info(f"Found {len(json_files)} JSON files in {era}")

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract publications - handle both dict with 'publications' key and direct list
                if isinstance(data, list):
                    publications = data
                elif isinstance(data, dict) and 'publications' in data:
                    publications = data['publications']
                else:
                    logger.warning(f"Unexpected JSON structure in {json_file.name}")
                    continue
                
                for pub in publications:
                    if not isinstance(pub, dict):
                        continue
                    
                    # Filter by year
                    if year_start and not self._matches_year(pub, year_start, year_end):
                        continue
                    
                    # Handle sections (Datamation/CACM style) or flat structure
                    if 'sections' in pub and isinstance(pub['sections'], list):
                        sections = pub['sections']
                    else:
                        sections = [pub]
                    
                    for section in sections:
                        if not isinstance(section, dict):
                            continue
                            
                        # Get text path (section takes precedence over pub)
                        text_path = section.get('text_path') or pub.get('text_path', '')
                        if not text_path:
                            continue
                        
                        # All paths should now be relative to CleanedSources root
                        full_path = self.cleaned_sources_path / text_path
                        if not full_path.exists():
                            logger.warning(f"Text file not found: {text_path}")
                            continue
                        
                        with open(full_path, 'r', encoding='utf-8') as f:
                            texts.append(f.read())
                        
                        # Build metadata
                        metadata = {
                            'era': era,
                            'source': pub.get('source', 'unknown'),
                            'publication_date': pub.get('publication_date', ''),
                            'title': section.get('section_title') or section.get('title') or pub.get('title', ''),
                            'filepath': str(text_path)
                        }
                        metadata_list.append(metadata)
                        
            except Exception as e:
                logger.warning(f"Error processing {json_file.name}: {e}")

        year_info = f" ({year_start}-{year_end})" if year_end else f" ({year_start})" if year_start else ""
        logger.info(f"Loaded {len(texts)} documents from {era}{year_info}")
        return texts, metadata_list

    def _matches_year(self, pub: Dict, year_start: int, year_end: Optional[int]) -> bool:
        """Check if publication matches year filter"""
        pub_date = pub.get('publication_date', '')
        if not pub_date:
            return True
        
        try:
            pub_year = int(pub_date.split('-')[0])
            if year_end:
                return year_start <= pub_year <= year_end
            return pub_year == year_start
        except (ValueError, IndexError):
            return True
