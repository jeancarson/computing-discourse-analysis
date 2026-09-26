#!/usr/bin/env python3
"""
Word Cloud Generator for Computing Discourse Data

Generates word clouds of the most popular words from cleaned data.
Prompts for era, year range, and data source. If not specified, includes all matching.
"""

import sys
from pathlib import Path
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent / 'bertopic'))

from core.data_loader import DataLoader

def get_user_input():
    """Prompt user for era, year, and source."""
    print("Word Cloud Generator")
    print("===================")

    # Era
    while True:
        era_input = input("Enter era (1 or 2, or press Enter for all): ").strip()
        if not era_input:
            eras = ['Era1', 'Era2']
            break
        elif era_input in ['1', '2']:
            eras = [f'Era{era_input}']
            break
        else:
            print("Invalid input. Enter 1, 2, or press Enter.")

    # Year
    year_start = None
    year_end = None
    year_input = input("Enter year or year range (e.g., 1970 or 1970-1975, or press Enter for all): ").strip()
    if year_input:
        if '-' in year_input:
            try:
                year_start, year_end = map(int, year_input.split('-'))
            except ValueError:
                print("Invalid year range. Using all years.")
        else:
            try:
                year_start = int(year_input)
            except ValueError:
                print("Invalid year. Using all years.")

    # Source
    source_input = input("Enter data source (e.g., CACM, Datamation, or press Enter for all): ").strip()
    source = source_input if source_input else None

    return eras, year_start, year_end, source

def load_stopwords(stopwords_file=None):
    """Load stopwords from file or use defaults."""
    if stopwords_file is None:
        stopwords_file = Path(__file__).parent / "stopwords.txt"
    
    if stopwords_file.exists():
        with open(stopwords_file, 'r') as f:
            stopwords = set(line.strip().lower() for line in f if line.strip())
    else:
        # Fallback to basic list
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'}
    return stopwords

def load_and_process_texts(eras, year_start, year_end, source):
    """Load texts based on user input and process into word list."""
    data_loader = DataLoader(Path(__file__).parent.parent.parent.parent / "data" / "cleaned")
    all_texts = []
    stopwords = load_stopwords()

    for era in eras:
        raw_texts, metadata_list = data_loader.load_era_data(era, year_start, year_end)
        if not raw_texts:
            continue

        # Filter by source if specified
        if source:
            filtered_texts = []
            for text, meta in zip(raw_texts, metadata_list):
                if meta.get('source', '').lower() == source.lower():
                    filtered_texts.append(text)
            raw_texts = filtered_texts

        all_texts.extend(raw_texts)

    if not all_texts:
        print("No texts found matching the criteria.")
        return []

    # Combine and tokenize
    combined_text = ' '.join(all_texts)
    words = re.findall(r'\b\w+\b', combined_text.lower())
    # Remove stopwords
    words = [word for word in words if word not in stopwords and len(word) > 2]
    return words

def generate_wordcloud(words, title):
    """Generate and display/save word cloud."""
    if not words:
        return

    word_freq = Counter(words)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

    # Save
    output_path = Path(__file__).parent / f"wordcloud_{title.replace(' ', '_').lower()}.png"
    wordcloud.to_file(output_path)
    print(f"Word cloud saved to {output_path}")

def main():
    eras, year_start, year_end, source = get_user_input()
    words = load_and_process_texts(eras, year_start, year_end, source)

    # Create title
    title_parts = []
    if len(eras) == 1:
        title_parts.append(eras[0])
    else:
        title_parts.append("All Eras")
    if year_start:
        if year_end:
            title_parts.append(f"{year_start}-{year_end}")
        else:
            title_parts.append(str(year_start))
    if source:
        title_parts.append(source)
    title = "Word Cloud: " + " - ".join(title_parts)

    generate_wordcloud(words, title)

if __name__ == "__main__":
    main()