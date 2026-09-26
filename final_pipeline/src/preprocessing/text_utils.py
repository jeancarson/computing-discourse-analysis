import re
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_lemmatizer = None
_nltk_stopwords = None


def get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def get_nltk_stopwords():
    global _nltk_stopwords
    if _nltk_stopwords is None:
        _nltk_stopwords = set(stopwords.words('english'))
    return _nltk_stopwords


def dehyphenate_text(text: str) -> str:
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"(?<=\w)-(?=\w)", "", text)
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def remove_special_characters(text: str) -> str:
    text = re.sub(r"[^A-Za-z\s\.\!\?]", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\s*([.!?])\s*", r"\1 ", text)
    return text.strip()


def lemmatize_text(text: str) -> str:
    lemmatizer = get_lemmatizer()
    lemmatized_chunks = []
    
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        sentences = re.split(r'[.!?]+', text)

    for sent in sentences:
        if not sent.strip():
            continue
        try:
            tokens = nltk.word_tokenize(sent)
        except Exception:
            tokens = sent.split()
        lemmatized = [lemmatizer.lemmatize(tok) for tok in tokens if tok.isalpha()]
        if lemmatized:
            lemmatized_chunks.append(' '.join(lemmatized))

    return '. '.join(lemmatized_chunks)


def is_number(token: str) -> bool:
    return any(c.isdigit() for c in token)


def remove_stopwords(text: str, custom_stopwords: List[str] = None) -> str:
    nltk_stops = get_nltk_stopwords()
    all_stopwords = set(nltk_stops)
    
    if custom_stopwords:
        all_stopwords.update(custom_stopwords)
    
    words = re.findall(r'\b\w+\b|[.!?]', text)
    filtered_words = []
    
    for word in words:
        if word in '.!?':
            if filtered_words and filtered_words[-1] not in '.!?':
                filtered_words.append(word)
        elif word.lower() not in all_stopwords and not is_number(word):
            filtered_words.append(word)
    
    return ' '.join(filtered_words)
