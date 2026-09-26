"""
Standardize all cleaned JSON files to have paths relative to CleanedSources root
"""

import json
from pathlib import Path


def standardize_json_file(json_path: Path, cleaned_sources_root: Path) -> None:
    """
    Standardize paths in JSON file to be relative to CleanedSources root
    """
    print(f"Processing: {json_path.relative_to(cleaned_sources_root)}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determine which era this file belongs to
    era = None
    if 'Era1' in str(json_path):
        era = 'Era1'
    elif 'Era2' in str(json_path):
        era = 'Era2'
    
    if not era:
        print(f"  Skipping - could not determine era")
        return
    
    # Extract publications
    if isinstance(data, list):
        publications = data
    elif 'publications' in data:
        publications = data['publications']
    else:
        print(f"  Skipping - no publications found")
        return
    
    # Standardize paths
    path_count = 0
    for pub in publications:
        # Handle nested sections
        if 'sections' in pub:
            for section in pub['sections']:
                if 'text_path' in section:
                    section['text_path'] = standardize_path(section['text_path'], era)
                    path_count += 1
        
        # Handle direct text_path
        if 'text_path' in pub:
            pub['text_path'] = standardize_path(pub['text_path'], era)
            path_count += 1
    
    # Write back
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  Standardized {path_count} paths")


def standardize_path(text_path: str, era: str) -> str:
    """
    Ensure path is relative to CleanedSources root (starts with Era1/ or Era2/)
    """
    if not text_path:
        return text_path
    
    # Remove leading slash
    text_path = text_path.lstrip('/')
    
    # Already starts with Era prefix - good
    if text_path.startswith('Era1/') or text_path.startswith('Era2/'):
        return text_path
    
    # Add era prefix
    return f"{era}/{text_path}"


def main():
    """Standardize all cleaned JSON files"""
    base_path = Path(__file__).parent.parent.parent.parent / "data" / "cleaned"
    
    # Find all JSON files
    json_files = list(base_path.rglob("*Cleaned.json"))
    
    print(f"Found {len(json_files)} JSON files to standardize\n")
    
    for json_file in json_files:
        try:
            standardize_json_file(json_file, base_path)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("\nStandardization complete!")


if __name__ == "__main__":
    main()
