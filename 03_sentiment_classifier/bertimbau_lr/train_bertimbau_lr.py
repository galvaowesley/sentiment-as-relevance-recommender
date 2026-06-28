"""
Train BERTimbau (frozen) + Logistic Regression sentiment classifier (Pipeline 1).

Steps:
    1. Load no-neutral train / val / test splits
    2. BERT-specific normalization (02_preprocessing/bert/tokenize.py)
    3. Extract [CLS] embeddings from frozen BERTimbau in batches (cached to disk)
    4. Stratified k=3 cross-validation with Grid Search on train embeddings
    5. Evaluation on val and test sets independently
    6. Save LR model + results to 03_sentiment_classifier/checkpoints/

Usage:
    python 03_sentiment_classifier/bertimbau_lr/train_bertimbau_lr.py
"""

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_preprocessing"))

from bert.tokenize import ReviewDataset, build_text_input_bert, load_tokenizer  # noqa: E402
from common.preprocess import map_polarity  # noqa: E402

warnings.filterwarnings("ignore", message="Unknown solver options: iprint")

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
BATCH_SIZE = 32
DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "03_sentiment_classifier" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name: str) -> tuple[pd.Series, pd.Series]:
    path = DATA_DIR / f"B2W-Reviews01_no_neutral_{name}.csv"
    df = pd.read_csv(path)
    X = build_text_input_bert(df)
    y = map_polarity(df)
    mask = y.notna()
    return X[mask].reset_index(drop=True), y[mask].astype(int).reset_index(drop=True)


def extract_cls_embeddings(
    texts: pd.Series,
    labels: pd.Series,
    tokenizer,
    model: AutoModel,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = ReviewDataset(texts, tokenizer, labels=labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    embeddings, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls)
            all_labels.append(batch["labels"].numpy())
            if (i + 1) % 20 == 0:
                print(f"    {i + 1}/{len(loader)} batches", end="\r")

    print()
    return np.vstack(embeddings), np.concatenate(all_labels)


def get_embeddings(
    name: str,
    texts: pd.Series,
    labels: pd.Series,
    tokenizer,
    model: AutoModel,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return [CLS] embeddings, loading from cache if available."""
    cache_X = CHECKPOINT_DIR / f"bert_cls_{name}_X.npy"
    cache_y = CHECKPOINT_DIR / f"bert_cls_{name}_y.npy"

    if cache_X.exists() and cache_y.exists():
        print(f"  Cache hit — loading {name} embeddings from disk.")
        return np.load(cache_X), np.load(cache_y)

    print(f"  Extracting {name} embeddings ({len(texts):,} samples)...")
    X, y = extract_cls_embeddings(texts, labels, tokenizer, model, device)
    np.save(cache_X, X)
    np.save(cache_y, y)
    print(f"  Saved to {cache_X.name} / {cache_y.name}")
    return X, y


def evaluate_split(model, X: np.ndarray, y: np.ndarray) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    f1 = f1_score(y, y_pred, average="macro")
    auc = roc_auc_score(y, y_proba)

    print(f"\n  F1-macro : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print("\n  Classification report:")
    print(classification_report(y, y_pred, target_names=["negative", "positive"]))
    print("  Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y, y_pred))

    return {"f1_macro": round(f1, 4), "auc_roc": round(auc, 4)}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 — Loading data")
    print("=" * 60)
    X_train_raw, y_train = load_split("train")
    X_val_raw, y_val = load_split("val")
    X_test_raw, y_test = load_split("test")

    print(f"  Train : {len(X_train_raw):,} samples")
    print(f"  Val   : {len(X_val_raw):,} samples")
    print(f"  Test  : {len(X_test_raw):,} samples")
    print(f"  Device: {device}")

    # ------------------------------------------------------------------
    # 2. Load BERTimbau (frozen)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — Loading BERTimbau (frozen weights)")
    print("=" * 60)
    tokenizer = load_tokenizer()
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    for param in model.parameters():
        param.requires_grad = False
    print(f"  Model : {MODEL_NAME}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,} (all frozen)")

    # ------------------------------------------------------------------
    # 3. Extract [CLS] embeddings
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 — Extracting [CLS] embeddings (batch_size={})".format(BATCH_SIZE))
    print("=" * 60)
    X_train, _ = get_embeddings("train", X_train_raw, y_train, tokenizer, model, device)
    X_val, _ = get_embeddings("val", X_val_raw, y_val, tokenizer, model, device)
    X_test, _ = get_embeddings("test", X_test_raw, y_test, tokenizer, model, device)
    print(f"  Embedding dim: {X_train.shape[1]}")

    # ------------------------------------------------------------------
    # 4. Grid Search with k=3 stratified cross-validation (train only)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 — Grid Search (k=3 StratifiedKFold, scoring=f1_macro)")
    print("=" * 60)
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "saga"],
        "max_iter": [1000],
    }
    lr = LogisticRegression(class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(
        lr,
        param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    print(f"\n  Best params       : {grid.best_params_}")
    print(f"  Best F1-macro (CV): {grid.best_score_:.4f}")

    best_model = grid.best_estimator_

    # ------------------------------------------------------------------
    # 5a. Evaluation on validation set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5a — Evaluation on validation set")
    print("=" * 60)
    val_metrics = evaluate_split(best_model, X_val, y_val.to_numpy())

    # ------------------------------------------------------------------
    # 5b. Evaluation on held-out test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5b — Evaluation on held-out test set")
    print("=" * 60)
    test_metrics = evaluate_split(best_model, X_test, y_test.to_numpy())

    # ------------------------------------------------------------------
    # 6. Save artifacts
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 — Saving artifacts")
    print("=" * 60)
    with open(CHECKPOINT_DIR / "bertimbau_lr_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    results = {
        "model": "bertimbau_lr",
        "bert_model": MODEL_NAME,
        "best_params": grid.best_params_,
        "cv_f1_macro": round(grid.best_score_, 4),
        "val": val_metrics,
        "test": test_metrics,
        "embedding_dim": int(X_train.shape[1]),
        "train_size": len(X_train_raw),
        "val_size": len(X_val_raw),
        "test_size": len(X_test_raw),
    }
    results_path = CHECKPOINT_DIR / "bertimbau_lr_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"  bertimbau_lr_model.pkl    → {CHECKPOINT_DIR}")
    print(f"  bertimbau_lr_results.json → {CHECKPOINT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
