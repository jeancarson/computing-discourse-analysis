"""
Domain-Specific Stopword Analysis Tool

This script analyzes your corpus to identify potential domain-specific stopwords using:
1. Most frequent terms (high raw frequency)
2. High document frequency (terms appearing in X% of documents)
3. Low IDF terms (inverse document frequency)

Usage:
    python analyze_stopwords.py --era 1                    # Analyze Era1
    python analyze_stopwords.py --era 2                    # Analyze Era2
    python analyze_stopwords.py --era 1 --top 50           # Show top 50 candidates
    python analyze_stopwords.py --era 1 --threshold 0.85   # Terms in 85%+ of docs
"""

import argparse
import logging
from pathlib import Path
from collections import Counter
import math

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk

from topic_modelling.config import load_config
from topic_modelling.core.data_loader import DataLoader
from preprocessing.text_preprocessor import TextPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StopwordAnalyzer:
    """Analyze corpus to identify potential domain-specific stopwords"""
    
    def __init__(self, texts, existing_stopwords=None):
        self.texts = texts
        self.existing_stopwords = set(existing_stopwords or stopwords.words('english'))
        self.n_docs = len(texts)
        
    def get_term_frequencies(self, normalize=True):
        """
        Calculate raw term frequencies across all documents
        
        Args:
            normalize: If True, normalize by document length
        
        Returns:
            Counter object with term frequencies
        """
        term_freq = Counter()
        
        for text in self.texts:
            words = text.lower().split()
            
            if normalize:
                # Normalize by document length
                doc_length = len(words)
                word_counts = Counter(words)
                for word, count in word_counts.items():
                    term_freq[word] += count / doc_length if doc_length > 0 else 0
            else:
                term_freq.update(words)
        
        return term_freq
    
    def get_document_frequencies(self):
        """
        Calculate document frequency for each term
        
        Returns:
            dict: {term: number of documents containing term}
        """
        doc_freq = Counter()
        
        for text in self.texts:
            unique_words = set(text.lower().split())
            doc_freq.update(unique_words)
        
        return doc_freq
    
    def get_idf_scores(self):
        """
        Calculate IDF (Inverse Document Frequency) for each term
        
        IDF = log(N / df) where N is total docs, df is doc frequency
        
        Returns:
            dict: {term: idf_score}
        """
        doc_freq = self.get_document_frequencies()
        idf_scores = {}
        
        for term, df in doc_freq.items():
            idf_scores[term] = math.log(self.n_docs / df)
        
        return idf_scores
    
    def analyze_high_frequency_terms(self, top_n=100, exclude_common=True):
        """
        Find most frequent terms that could be stopwords
        
        Args:
            top_n: Number of top terms to return
            exclude_common: If True, exclude standard English stopwords
        
        Returns:
            list of (term, frequency) tuples
        """
        term_freq = self.get_term_frequencies(normalize=True)
        
        if exclude_common:
            # Remove already-known stopwords
            term_freq = {k: v for k, v in term_freq.items() 
                        if k not in self.existing_stopwords}
            term_freq = Counter(term_freq)
        
        return term_freq.most_common(top_n)
    
    def analyze_high_df_terms(self, threshold=0.85, exclude_common=True):
        """
        Find terms appearing in X% or more of documents
        
        Args:
            threshold: Percentage threshold (0.0 to 1.0)
            exclude_common: If True, exclude standard English stopwords
        
        Returns:
            list of (term, doc_count, percentage) tuples
        """
        doc_freq = self.get_document_frequencies()
        
        results = []
        for term, count in doc_freq.items():
            percentage = count / self.n_docs
            
            if percentage >= threshold:
                if not exclude_common or term not in self.existing_stopwords:
                    results.append((term, count, percentage))
        
        # Sort by percentage descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def analyze_low_idf_terms(self, top_n=100, exclude_common=True):
        """
        Find terms with lowest IDF scores (appear in many documents)
        
        Args:
            top_n: Number of bottom terms to return
            exclude_common: If True, exclude standard English stopwords
        
        Returns:
            list of (term, idf_score) tuples
        """
        idf_scores = self.get_idf_scores()
        
        if exclude_common:
            idf_scores = {k: v for k, v in idf_scores.items() 
                         if k not in self.existing_stopwords}
        
        # Sort by IDF ascending (lowest first)
        sorted_terms = sorted(idf_scores.items(), key=lambda x: x[1])
        return sorted_terms[:top_n]
    
    def analyze_rare_terms(self, min_count=1):
        """
        Find extremely rare terms (potential noise)
        
        Args:
            min_count: Minimum term frequency threshold
        
        Returns:
            list of (term, count) tuples
        """
        term_freq = self.get_term_frequencies(normalize=False)
        rare_terms = [(term, count) for term, count in term_freq.items() 
                     if count <= min_count]
        return sorted(rare_terms, key=lambda x: x[1])


def format_results_table(results, columns, title):
    """Format results as a nice table"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    df = pd.DataFrame(results, columns=columns)
    print(df.to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze corpus for domain-specific stopwords')
    parser.add_argument('--era', type=int, choices=[1, 2], required=True, 
                       help='Era to analyze (1 or 2)')
    parser.add_argument('--top', type=int, default=50, 
                       help='Number of top candidates to show (default: 50)')
    parser.add_argument('--threshold', type=float, default=0.85, 
                       help='Document frequency threshold (0.0-1.0, default: 0.85)')
    parser.add_argument('--output', type=str, 
                       help='Optional: Save results to CSV file')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    config = load_config(script_dir.parent / "bertopic" / "config.json")
    data_loader = DataLoader(script_dir.parent.parent.parent / "data" / "cleaned")
    preprocessor = TextPreprocessor(config)
    
    # Load data
    era = f'Era{args.era}'
    logger.info(f"Loading {era} documents...")
    raw_texts, metadata_list = data_loader.load_era_data(era)
    
    if not raw_texts:
        logger.error(f"No documents found for {era}")
        return
    
    logger.info(f"Loaded {len(raw_texts)} documents")
    
    # Preprocess (but keep stopwords for analysis)
    logger.info("Preprocessing texts (keeping stopwords for analysis)...")
    
    # Temporarily disable stopword removal
    original_setting = config.preprocessing.remove_stopwords
    config.preprocessing.remove_stopwords = False
    
    preprocessor_no_stop = TextPreprocessor(config)
    processed_texts = [preprocessor_no_stop.preprocess_text(text) for text in raw_texts]
    
    # Restore original setting
    config.preprocessing.remove_stopwords = original_setting
    
    # Initialize analyzer
    logger.info("Analyzing corpus...")
    analyzer = StopwordAnalyzer(processed_texts, config.vectorizer.additional_stopwords)
    
    # Analysis 1: High frequency terms
    logger.info(f"\n1. Analyzing top {args.top} most frequent terms...")
    high_freq = analyzer.analyze_high_frequency_terms(top_n=args.top)
    format_results_table(
        high_freq[:20],  # Show top 20
        ['Term', 'Normalized Frequency'],
        f"Top 20 Most Frequent Terms (excluding common English stopwords)"
    )
    
    # Analysis 2: High document frequency
    logger.info(f"\n2. Analyzing terms appearing in {args.threshold*100}%+ of documents...")
    high_df = analyzer.analyze_high_df_terms(threshold=args.threshold)
    format_results_table(
        high_df[:20],
        ['Term', 'Doc Count', 'Percentage'],
        f"Terms Appearing in {args.threshold*100}%+ of Documents"
    )
    
    # Analysis 3: Low IDF scores
    logger.info(f"\n3. Analyzing terms with lowest IDF scores...")
    low_idf = analyzer.analyze_low_idf_terms(top_n=args.top)
    format_results_table(
        low_idf[:20],
        ['Term', 'IDF Score'],
        "Terms with Lowest IDF Scores (appear in many documents)"
    )
    
    # Analysis 4: Rare terms (potential noise)
    logger.info("\n4. Analyzing rare terms (appearing only once)...")
    rare_terms = analyzer.analyze_rare_terms(min_count=1)
    print(f"\nFound {len(rare_terms)} terms appearing only once")
    print(f"Examples: {', '.join([t[0] for t in rare_terms[:20]])}")
    
    # Generate recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR CUSTOM STOPWORDS")
    print(f"{'='*80}")
    
    # Combine insights
    high_freq_terms = set(term for term, _ in high_freq[:30])
    high_df_terms = set(term for term, _, _ in high_df[:30])
    low_idf_terms = set(term for term, _ in low_idf[:30])
    
    # Terms appearing in multiple analyses are strong candidates
    strong_candidates = high_freq_terms & high_df_terms & low_idf_terms
    medium_candidates = (high_freq_terms & high_df_terms) | (high_freq_terms & low_idf_terms) | (high_df_terms & low_idf_terms)
    medium_candidates -= strong_candidates
    
    print("\n🔴 STRONG CANDIDATES (appear in all 3 analyses):")
    print(f"   {', '.join(sorted(strong_candidates))}")
    
    print("\n🟡 MEDIUM CANDIDATES (appear in 2 analyses):")
    print(f"   {', '.join(sorted(medium_candidates))}")
    
    print("\n💡 TIPS:")
    print("   1. Add STRONG candidates to your stopword list")
    print("   2. Review MEDIUM candidates - add if they're generic/uninformative")
    print("   3. Keep domain-specific technical terms even if frequent")
    print("   4. Consider context - some common words may be topically important")
    
    # Save results if requested
    if args.output:
        output_path = script_dir / args.output
        results_df = pd.DataFrame({
            'Term': [t for t, _ in high_freq],
            'Normalized_Freq': [f for _, f in high_freq],
            'Doc_Percentage': [
                next((p for term, _, p in high_df if term == t), None) 
                for t, _ in high_freq
            ],
            'IDF_Score': [
                next((score for term, score in low_idf if term == t), None) 
                for t, _ in high_freq
            ]
        })
        results_df.to_csv(output_path, index=False)
        logger.info(f"\nResults saved to: {output_path}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
