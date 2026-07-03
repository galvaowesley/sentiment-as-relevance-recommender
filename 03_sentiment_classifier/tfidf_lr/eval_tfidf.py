"""
Inference script for TF-IDF + Logistic Regression.

Steps:
    1. Load validation and test splits (including original DataFrames).
    2. Concatenate DataFrames and raw texts.
    3. Lemmatize the combined text.
    4. Load the trained TF-IDF Vectorizer and Logistic Regression model.
    5. Transform texts and perform inference.
    6. Evaluate metrics.
    7. Save a copy of the original dataset with the new 'inferencia_tfidf' column.

Usage:
    python 04_inference/inference_tfidf_lr.py --ngrams both
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

# Configuração de caminhos
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_preprocessing"))

from common.preprocess import build_text_input, map_polarity  # noqa: E402
from tfidf.vectorize import lemmatize_series  # noqa: E402

warnings.filterwarnings("ignore", message="Unknown solver options: iprint")

DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "03_sentiment_classifier" / "checkpoints"
OUTPUT_DIR = ROOT / "03_sentiment_classifier" / "inference" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name: str, neutral:str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Carrega o split, mapeia a polaridade e retorna o DataFrame original filtrado junto com X e y."""
    if neutral == "":
        path = DATA_DIR / f"B2W-Reviews01_{name}.csv"
    else:
        path = DATA_DIR / f"B2W-Reviews01_no_neutral_{name}.csv"
    df = pd.read_csv(path)
    X = build_text_input(df)
    y = map_polarity(df)
    
    # Criar máscara para remover nulos
    mask = y.notna()
    
    # Retorna o DF original filtrado, além de X e y
    df_filtered = df[mask].reset_index(drop=True)
    X_filtered = X[mask].reset_index(drop=True)
    y_filtered = y[mask].astype(int).reset_index(drop=True)
    
    return df_filtered, X_filtered, y_filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for TF-IDF + Logistic Regression.")
    parser.add_argument(
        "--ngrams",
        type=str,
        choices=["unigrams", "bigrams", "both"],
        default="both",
        help="Estratégia de extração utilizada no treinamento."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["no_neutral", ""],
        default="",
        help="Dataset com ou sem neutros."
    )
    args = parser.parse_args()
    prefix = f"tfidf_{args.ngrams}"

    print("Carregando datasets de validação e teste...")
    df_val, X_val_raw, y_val = load_split("val", args.dataset)
    df_test, X_test_raw, y_test = load_split("test", args.dataset)

    # Juntando validação e teste
    df_combined = pd.concat([df_val, df_test]).reset_index(drop=True)
    X_combined_raw = pd.concat([X_val_raw, X_test_raw]).reset_index(drop=True)
    y_combined = pd.concat([y_val, y_test]).reset_index(drop=True)
    
    print(f"  Amostras carregadas : {len(df_combined):,}\n")

    X_combined_lemmatized = lemmatize_series(X_combined_raw)
    
    print(f"Carregando modelos (Estratégia: {args.ngrams})...")
    vectorizer_path = CHECKPOINT_DIR / f"{prefix}_vectorizer.pkl"
    lr_model_path = CHECKPOINT_DIR / f"{prefix}_lr_model.pkl"

    if not vectorizer_path.exists() or not lr_model_path.exists():
        raise FileNotFoundError(f"Modelos não encontrados no diretório: {CHECKPOINT_DIR}")

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(lr_model_path, "rb") as f:
        lr_model = pickle.load(f)
    
    X_embeddings = vectorizer.transform(X_combined_lemmatized)
    print(f"  Shape final: {X_embeddings.shape}\n")

    y_pred = lr_model.predict(X_embeddings)
    y_proba = lr_model.predict_proba(X_embeddings)[:, 1]

    f1 = f1_score(y_combined, y_pred, average="macro")
    auc = roc_auc_score(y_combined, y_proba)
    report_dict = classification_report(y_combined, y_pred, target_names=["negative", "positive"], output_dict=True)

    print(f"  F1-macro : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")

    
    # Adicionando atributos à cópia do dataset original
    df_combined["inferencia_tfidf"] = y_pred
    df_combined["probabilidade_tfidf"] = np.round(y_proba, 4)
    
    # Exportando dataset
    csv_path = OUTPUT_DIR / f"B2W-Reviews01_inferred_{prefix}.csv"
    df_combined.to_csv(csv_path, index=False)
    
    # Exportando métricas
    metrics = {
        "dataset_size": len(df_combined),
        "ngram_strategy": args.ngrams,
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
    json_path = OUTPUT_DIR / f"metrics_inferred_{prefix}.json"
    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"  Dataset salvo em: {csv_path}")
    print(f"  Métricas salvas em: {json_path}")
    print("\nConcluído.")


if __name__ == "__main__":
    main()