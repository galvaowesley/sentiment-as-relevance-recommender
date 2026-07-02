"""
Train TF-IDF + Logistic Regression sentiment classifier (Pipeline 1 — baseline).

Steps:
    1. Load no-neutral train / val / test splits
    2. Common text normalization  (02_preprocessing/common/preprocess.py)
    3. Lemmatization + TF-IDF vectorization (02_preprocessing/tfidf/vectorize.py)
    4. Stratified k=5 cross-validation with Grid Search on train+val combined
    5. Final evaluation on held-out test set
    6. Save vectorizer + model to 03_sentiment_classifier/checkpoints/

Usage:
    python 03_sentiment_classifier/tfidf_lr/train_tfidf.py --ngrams both
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_preprocessing"))

from common.preprocess import build_text_input, map_polarity  # noqa: E402
from tfidf.vectorize import build_vectorizer, lemmatize_series  # noqa: E402

warnings.filterwarnings("ignore", message="Unknown solver options: iprint")

DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "03_sentiment_classifier" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name: str) -> tuple[pd.Series, pd.Series]:
    path = DATA_DIR / f"B2W-Reviews01_no_neutral_{name}.csv"
    df = pd.read_csv(path)
    X = build_text_input(df)
    y = map_polarity(df)
    mask = y.notna()
    return X[mask].reset_index(drop=True), y[mask].astype(int).reset_index(drop=True)


def evaluate_split(model, X_tfidf, y_true) -> dict:
    y_pred = model.predict(X_tfidf)
    y_proba = model.predict_proba(X_tfidf)[:, 1]

    f1 = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_proba)

    print(f"\n  F1-macro : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print("\n  Classification report:")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))
    
    report_dict = classification_report(y_true, y_pred, target_names=["negative", "positive"], output_dict=True)
    
    print("  Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))

    return {
        "f1_macro": round(f1, 4),
        "auc_roc": round(auc, 4),
        "classes": {
            "negative": {
                "precision": round(report_dict["negative"]["precision"], 4),
                "recall": round(report_dict["negative"]["recall"], 4),
                "f1_score": round(report_dict["negative"]["f1-score"], 4),
            },
            "positive": {
                "precision": round(report_dict["positive"]["precision"], 4),
                "recall": round(report_dict["positive"]["recall"], 4),
                "f1_score": round(report_dict["positive"]["f1-score"], 4),
            }
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression sentiment classifier.")
    parser.add_argument(
        "--ngrams",
        type=str,
        choices=["unigrams", "bigrams", "both"],
        default="both",
        help="Estratégia de extração: 'unigrams' (1,1), 'bigrams' (2,2) ou 'both' (1,2)."
    )
    args = parser.parse_args()

    ngram_mapping = {
        "unigrams": (1, 1),
        "bigrams": (2, 2),
        "both": (1, 2)
    }
    selected_ngram_range = ngram_mapping[args.ngrams]

    print("=" * 60)
    print("STEP 1 — Loading data")
    print("=" * 60)
    X_train_raw, y_train = load_split("train")
    X_val_raw, y_val = load_split("val")
    X_test_raw, y_test = load_split("test")

    print(f"  Train : {len(X_train_raw):,} samples")
    print(f"  Val   : {len(X_val_raw):,} samples")
    print(f"  Test  : {len(X_test_raw):,} samples")
    dist = y_train.value_counts().sort_index()
    print(f"  Class dist (train): neg={dist[0]:,}  pos={dist[1]:,}")

    print("\n" + "=" * 60)
    print("STEP 2 — Lemmatizing texts (spaCy pt_core_news_sm)")
    print("=" * 60)
    X_train = lemmatize_series(X_train_raw)
    X_val = lemmatize_series(X_val_raw)
    X_test = lemmatize_series(X_test_raw)
    print("  Done.")

    print("\n" + "=" * 60)
    print(f"STEP 3 — TF-IDF vectorization ({args.ngrams}, max 50k features)")
    print("=" * 60)
    
    # Passando dinamicamente a tupla selecionada
    vectorizer = build_vectorizer(ngram_range=selected_ngram_range)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"  Vocabulary size : {len(vectorizer.vocabulary_):,}")
    print(f"  Train shape     : {X_train_tfidf.shape}")
    print(f"  Val shape       : {X_val_tfidf.shape}")
    print(f"  Test shape      : {X_test_tfidf.shape}")

    print("\n" + "=" * 60)
    print("STEP 4 — Grid Search (k=5 StratifiedKFold, scoring=f1_macro)")
    print("=" * 60)
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "saga"],
        "max_iter": [1000],
    }
    unbalanced = {0: 2.5, 1: 1.0}
    lr = LogisticRegression(class_weight=unbalanced, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        lr,
        param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_train_tfidf, y_train)

    print(f"\n  Best params       : {grid.best_params_}")
    print(f"  Best F1-macro (CV): {grid.best_score_:.4f}")

    best_model = grid.best_estimator_

    print("\n" + "=" * 60)
    print("STEP 5a — Evaluation on validation set")
    print("=" * 60)
    val_metrics = evaluate_split(best_model, X_val_tfidf, y_val)

    
    print("\n" + "=" * 60)
    print("STEP 5b — Evaluation on held-out test set")
    print("=" * 60)
    test_metrics = evaluate_split(best_model, X_test_tfidf, y_test)

    # 6. Salvando

    print("\n" + "=" * 60)
    print("STEP 6 — Saving artifacts")
    print("=" * 60)
    
    # Adicionando o tipo de n-grama ao nome dos arquivos para evitar sobrescrever
    prefix = f"tfidf_{args.ngrams}"
    
    with open(CHECKPOINT_DIR / f"{prefix}_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(CHECKPOINT_DIR / f"{prefix}_lr_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    results = {
        "model": f"tfidf_lr_{args.ngrams}",
        "ngram_strategy": args.ngrams,
        "ngram_range": selected_ngram_range,
        "best_params": grid.best_params_,
        "cv_f1_macro": round(grid.best_score_, 4),
        "val": val_metrics,
        "test": test_metrics,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "train_size": len(X_train_raw),
        "val_size": len(X_val_raw),
        "test_size": len(X_test_raw),
    }
    
    results_path = CHECKPOINT_DIR / f"{prefix}_lr_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"  {prefix}_vectorizer.pkl  → {CHECKPOINT_DIR}")
    print(f"  {prefix}_lr_model.pkl    → {CHECKPOINT_DIR}")
    print(f"  {prefix}_lr_results.json → {CHECKPOINT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()