"""
Common text normalization applied to all models (TF-IDF and BERTimbau).

Input:  review_title + review_text columns from the B2W-Reviews01 corpus.
Output: a single normalized string per row.
"""

import re
import unicodedata

import pandas as pd


def normalize_text(text: str) -> str:
    """Apply base normalization to a single string.

    Steps (applied in order):
        1. Lowercase
        2. Remove HTML tags
        3. Remove URLs
        4. Unicode normalization to NFC (consistent encoding)
        5. Remove non-alphanumeric characters (keep letters, digits, spaces)
        6. Collapse duplicate whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^a-záàâãéèêíïóôõöúüçñ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_text_input(df: pd.DataFrame) -> pd.Series:
    """Concatenate review_title and review_text into a single input field."""
    title = df["review_title"].fillna("").astype(str)
    body = df["review_text"].fillna("").astype(str)
    combined = (title + " " + body).str.strip()
    return combined.apply(normalize_text)


def map_polarity(df: pd.DataFrame) -> pd.Series:
    """Map overall_rating to a binary sentiment label.

    Returns:
        0 for negative (ratings 1–2)
        1 for positive (ratings 4–5)
        NaN for neutral (rating 3) — caller should drop these rows.
    """
    mapping = {1: 0, 2: 0, 4: 1, 5: 1}
    return df["overall_rating"].map(mapping)
