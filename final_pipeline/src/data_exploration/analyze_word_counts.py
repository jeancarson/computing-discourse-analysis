import os
import json
import glob
from pathlib import Path

def count_words(text):
    return len(text.split())

def analyze_cleaned_sources(base_dir):
    word_counts = []
    short_files = []
    long_files = []
    for era in ['Era1', 'Era2']:
        era_path = os.path.join(base_dir, era)
        if not os.path.exists(era_path):
            continue
        # Find all .txt files in this era
        for root, dirs, files in os.walk(era_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        wc = count_words(text)
                        word_counts.append(wc)
                        rel_path = os.path.relpath(file_path, base_dir)
                        if wc < 50:
                            short_files.append(rel_path)
                        elif wc > 6000:
                            long_files.append((rel_path, wc))
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    if not word_counts:
        print("No word counts found.")
        return
    print(f"Total text files: {len(word_counts)}")
    print(f"Min words: {min(word_counts)}")
    print(f"Max words: {max(word_counts)}")
    print(f"Average words: {sum(word_counts)/len(word_counts):.2f}")
    print(f"\nFiles with less than 50 words ({len(short_files)}):")
    for f in short_files:
        print(f"  {f}")
    print(f"\nFiles with more than 6000 words ({len(long_files)}):")
    for f, wc in long_files:
        print(f"  {f}: {wc} words")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    base_dir = str(project_root / "Sources" / "CleanedSources")
    analyze_cleaned_sources(base_dir)
