"""
Utility functions for sentiment analysis
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)


def load_documents_for_sentiment(data_root: Path, era: int) -> Tuple[List[str], List[str], List[str], List[Dict]]:
    """
    Load documents for sentiment analysis from the cleaned directory.
    
    Uses original_filepath from the index to read each document from its 
    unique location, avoiding the filename collision issue in the processed folder.
    
    Args:
        data_root: Path to data directory (e.g., Path('../../data'))
        era: Era number (1 or 2)
    
    Returns:
        Tuple of (texts, doc_ids, publication_dates, metadata_list)
    """
    era_str = f'Era{era}'
    processed_dir = data_root / 'processed' / era_str
    cleaned_dir = data_root / 'cleaned'
    index_file = processed_dir / f"{era_str}_index.json"
    
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    with open(index_file, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    texts = []
    doc_ids = []
    publication_dates = []
    metadata_list = []
    
    for item in index_data:
        # Use original_filepath to get the actual file path with full directory structure
        # This ensures each document is read from its unique location in cleaned/
        if 'original_filepath' in item:
            text_path = cleaned_dir / item['original_filepath']
        elif 'filepath' in item.get('metadata', {}):
            text_path = cleaned_dir / item['metadata']['filepath']
        else:
            # Fallback to old behavior
            text_path = processed_dir / item.get("text_file", item.get("chunk_file", ""))
        
        if not text_path.exists():
            logger.warning(f"Text file not found: {text_path}")
            continue
        
        with open(text_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
        
        # Extract metadata
        metadata = item.get('metadata', {})
        doc_id = item.get('original_filepath', metadata.get('filepath', text_path.stem))
        pub_date = metadata.get('publication_date', '')
        
        doc_ids.append(doc_id)
        publication_dates.append(pub_date)
        metadata_list.append(metadata)
    
    logger.info(f"Loaded {len(texts)} documents for sentiment analysis from Era{era}")
    logger.info(f"Documents with dates: {sum(1 for d in publication_dates if d)}")
    
    return texts, doc_ids, publication_dates, metadata_list


def convert_sentiment_to_numeric(label: str, score: float, compound_score: float = None) -> float:
    """
    Convert sentiment to a numeric score on [-1, +1].
    
    Preferred approach: Use compound_score (= P(positive) - P(negative))
    which is computed directly from the model's full probability distribution.
    This gives a continuous score where even 'neutral' documents have a
    non-zero value reflecting their slight lean towards positive or negative.
    
    Fallback: If compound_score is not available (legacy data), use
    label-based mapping: positive -> +score, negative -> -score, 
    neutral -> 0.0.
    
    Args:
        label: Sentiment label ('positive', 'neutral', 'negative')
        score: Confidence score for the top class (0-1)
        compound_score: P(positive) - P(negative), range [-1, +1]. 
                        If provided, this is returned directly.
    
    Returns:
        Numeric sentiment score in [-1, +1]
    
    Example:
        >>> convert_sentiment_to_numeric('positive', 0.9, compound_score=0.85)
        0.85
        >>> convert_sentiment_to_numeric('neutral', 0.75, compound_score=0.03)
        0.03
        >>> convert_sentiment_to_numeric('negative', 0.8)  # legacy fallback
        -0.8
    """
    # Prefer compound_score when available (new pipeline output)
    if compound_score is not None:
        return float(compound_score)
    
    # Legacy fallback for old data without compound_score
    label_lower = label.lower()
    
    if label_lower == 'positive':
        return 1.0 * score
    elif label_lower == 'negative':
        return -1.0 * score
    else:  # neutral
        return 0.0


def chunk_text_by_sentences(text: str, max_words: int = 400) -> List[str]:
    """
    Split text into chunks by sentences to stay within token limit
    
    RoBERTa has a 512 token limit. We use ~400 words as a safe threshold
    to account for subword tokenization.
    
    Args:
        text: Input text to chunk
        max_words: Maximum words per chunk (default: 400)
    
    Returns:
        List of text chunks
    
    Example:
        >>> text = "First sentence. Second sentence. Third sentence."
        >>> chunks = chunk_text_by_sentences(text, max_words=3)
        >>> len(chunks)
        3
    """
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        
        # If adding this sentence exceeds limit, save current chunk and start new one
        if current_word_count + sentence_word_count > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_word_count
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks if chunks else [text]  # Return original if chunking fails


def aggregate_chunk_sentiments(chunk_sentiments: List[Dict]) -> Dict:
    """
    Aggregate sentiments from multiple chunks using probability averaging.
    
    When full probability distributions are available (prob_positive, 
    prob_neutral, prob_negative), averages each probability across chunks
    and recomputes compound_score = mean(P(pos)) - mean(P(neg)).
    
    Falls back to majority voting for label if probabilities are not present.
    
    Args:
        chunk_sentiments: List of sentiment dictionaries with 'label', 'score',
                         and optionally probability fields
    
    Returns:
        Aggregated sentiment with label, score, compound_score, and probabilities
    
    Example:
        >>> chunks = [
        ...     {'label': 'neutral', 'score': 0.7, 'compound_score': 0.05,
        ...      'prob_positive': 0.15, 'prob_neutral': 0.7, 'prob_negative': 0.15},
        ...     {'label': 'neutral', 'score': 0.6, 'compound_score': -0.10,
        ...      'prob_positive': 0.10, 'prob_neutral': 0.6, 'prob_negative': 0.30},
        ... ]
        >>> result = aggregate_chunk_sentiments(chunks)
        >>> result['compound_score']  # (0.05 + -0.10) / 2 = -0.025
        -0.025
    """
    if not chunk_sentiments:
        return {
            'label': 'neutral', 'score': 0.5, 'compound_score': 0.0,
            'prob_positive': 0.0, 'prob_neutral': 1.0, 'prob_negative': 0.0
        }
    
    # Check if we have full probability distributions
    has_probs = 'prob_positive' in chunk_sentiments[0]
    
    if has_probs:
        # Average the full probability distributions across chunks
        avg_p_pos = np.mean([s['prob_positive'] for s in chunk_sentiments])
        avg_p_neu = np.mean([s['prob_neutral'] for s in chunk_sentiments])
        avg_p_neg = np.mean([s['prob_negative'] for s in chunk_sentiments])
        
        # Compound score from averaged probabilities
        compound_score = float(avg_p_pos - avg_p_neg)
        
        # Top label from averaged probabilities
        probs = {'positive': avg_p_pos, 'neutral': avg_p_neu, 'negative': avg_p_neg}
        final_label = max(probs, key=probs.get)
        final_score = probs[final_label]
        
        return {
            'label': final_label,
            'score': float(final_score),
            'compound_score': compound_score,
            'prob_positive': float(avg_p_pos),
            'prob_neutral': float(avg_p_neu),
            'prob_negative': float(avg_p_neg)
        }
    else:
        # Legacy fallback: majority vote for label, mean for score
        labels = [s['label'] for s in chunk_sentiments]
        final_label = max(set(labels), key=labels.count)
        final_score = np.mean([s['score'] for s in chunk_sentiments])
        
        return {
            'label': final_label,
            'score': float(final_score)
        }


def calculate_sentiment_statistics(sentiments: List[Dict]) -> Dict:
    """
    Calculate statistical summary of sentiment distribution.
    
    Uses compound_score (= P(positive) - P(negative)) when available,
    which gives continuous values even for all-neutral topics. Falls
    back to convert_sentiment_to_numeric for legacy data.
    
    Args:
        sentiments: List of sentiment dictionaries
    
    Returns:
        Dictionary with statistical measures
    """
    if not sentiments:
        return {
            'mean_sentiment': 0.0,
            'std_sentiment': 0.0,
            'positive_pct': 0.0,
            'neutral_pct': 0.0,
            'negative_pct': 0.0,
            'count': 0
        }
    
    # Convert to numeric scores — prefer compound_score if available
    numeric_scores = [
        convert_sentiment_to_numeric(
            s['label'], s['score'],
            compound_score=s.get('compound_score', None)
        )
        for s in sentiments
    ]
    
    # Count labels
    total = len(sentiments)
    positive_count = sum(1 for s in sentiments if s['label'].lower() == 'positive')
    negative_count = sum(1 for s in sentiments if s['label'].lower() == 'negative')
    neutral_count = sum(1 for s in sentiments if s['label'].lower() == 'neutral')
    
    return {
        'mean_sentiment': float(np.mean(numeric_scores)),
        'std_sentiment': float(np.std(numeric_scores)),
        'positive_pct': positive_count / total,
        'neutral_pct': neutral_count / total,
        'negative_pct': negative_count / total,
        'count': total
    }
