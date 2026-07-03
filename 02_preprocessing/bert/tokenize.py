"""
BERT-specific preprocessing for BERTimbau (neuralmind/bert-base-portuguese-cased).

Unlike the TF-IDF pipeline, this module applies a lighter normalization:
  - No lowercasing  (model is case-sensitive)
  - No accent removal  (accents carry lexical meaning in Portuguese)
  - HTML tags and URLs are still stripped
  - Whitespace is collapsed

The tokenizer handles subword splitting (WordPiece) internally.

Usage:
    from bert.tokenize import build_text_input_bert, load_tokenizer, ReviewDataset
"""

import re
import unicodedata

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 256  # covers >99% of B2W reviews; 512 is the hard BERT limit


def normalize_text_bert(text: str) -> str:
    """Strip noise without altering case or accents."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_text_input_bert(df: pd.DataFrame) -> pd.Series:
    """Concatenate review_title and review_text, apply BERT normalization."""
    title = df["review_title"].fillna("").astype(str)
    body = df["review_text"].fillna("").astype(str)
    combined = (title + " " + body).str.strip()
    return combined.apply(normalize_text_bert)


def load_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)


class ReviewDataset(Dataset):
    """PyTorch Dataset for B2W reviews tokenized for BERTimbau.

    Tokenizes the full series up front and stores tensors in memory.
    Pass labels=None for inference — __getitem__ will not include a 'labels' key.
    """

    def __init__(
        self,
        texts: pd.Series,
        tokenizer: AutoTokenizer,
        labels: pd.Series | None = None,
        max_length: int = MAX_LENGTH,
    ) -> None:
        encodings = tokenizer(
            texts.tolist(),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = (
            torch.tensor(labels.tolist(), dtype=torch.long)
            if labels is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item
