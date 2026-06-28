"""
TF-IDF specific preprocessing for the sparse baseline model.

Applies on top of the common normalization from 02_preprocessing/common/preprocess.py:
    1. Tokenization
    2. Portuguese stopword removal (NLTK)
    3. Lemmatization (spaCy pt_core_news_sm)
    4. TF-IDF vectorization with uni+bigrams

Usage:
    from 02_preprocessing.tfidf.vectorize import lemmatize_series, build_vectorizer
"""

import re

import nltk
import pandas as pd
import spacy
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

_PT_STOPWORDS = set(nltk_stopwords.words("portuguese"))


def _load_spacy():
    try:
        return spacy.load("pt_core_news_sm", disable=["parser", "ner"])
    except OSError as e:
        raise OSError(
            "spaCy model not found. Run: python -m spacy download pt_core_news_sm"
        ) from e


def lemmatize_text(text: str, nlp, stopwords: set) -> str:
    """Tokenize, remove stopwords and lemmatize a single string."""
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in tqdm.tqdm(doc, desc="Lemmatizing texts")
        if token.is_alpha and token.lemma_.lower() not in stopwords
    ]
    return " ".join(tokens)


def lemmatize_series(texts: pd.Series, batch_size: int = 512) -> pd.Series:
    """Apply lemmatization to a pandas Series using spaCy pipe for efficiency."""
    nlp = _load_spacy()
    stopwords = _PT_STOPWORDS

    results = []
    for doc in tqdm.tqdm(nlp.pipe(texts.astype(str), batch_size=batch_size), desc="Lemmatizing texts"):
        tokens = [
            token.lemma_
            for token in doc
            if token.is_alpha and token.lemma_.lower() not in stopwords
        ]
        results.append(" ".join(tokens))

    return pd.Series(results, index=texts.index)


def build_vectorizer(
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    min_df: int = 2,
) -> TfidfVectorizer:
    """Return a configured TfidfVectorizer ready to fit on lemmatized texts.

    Args:
        max_features: Maximum number of features (vocabulary size).
        ngram_range: Range of n-gram sizes (default: uni and bigrams).
        sublinear_tf: Apply log(1 + tf) scaling to term frequencies.
        min_df: Ignore terms appearing in fewer than min_df documents.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        analyzer="word",
        strip_accents=None,
    )
