"""
Inference script for BERTimbau + Logistic Regression.

Steps:
    1. Load validation and test splits (including original DataFrames).
    2. Concatenate DataFrames and features.
    3. Load the trained Logistic Regression model and frozen BERTimbau.
    4. Extract [CLS] embeddings for the concatenated dataset.
    5. Perform inference and evaluate.
    6. Save a copy of the original dataset with the new 'inferencia_bertimbau' column.
"""

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModel

# Configuração de caminhos
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_preprocessing"))

from bert.tokenize import ReviewDataset, build_text_input_bert, load_tokenizer  # noqa: E402
from common.preprocess import map_polarity  # noqa: E402

warnings.filterwarnings("ignore", message="Unknown solver options: iprint")

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
BATCH_SIZE = 32
DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "03_sentiment_classifier" / "checkpoints"
OUTPUT_DIR = ROOT / "03_sentiment_classifier" / "inference" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Carrega o split, aplica preprocessamento e retorna o DataFrame original filtrado junto com X e y."""
    path = DATA_DIR / f"B2W-Reviews01_no_neutral_{name}.csv"
    df = pd.read_csv(path)
    X = build_text_input_bert(df)
    y = map_polarity(df)
    
    # Criar máscara para remover nulos
    mask = y.notna()
    
    # Retornamos o DF original filtrado, além de X e y
    df_filtered = df[mask].reset_index(drop=True)
    X_filtered = X[mask].reset_index(drop=True)
    y_filtered = y[mask].astype(int).reset_index(drop=True)
    
    return df_filtered, X_filtered, y_filtered


def extract_cls_embeddings(
    texts: pd.Series,
    labels: pd.Series,
    tokenizer,
    model: AutoModel,
    device: torch.device,
) -> np.ndarray:
    """Extrai os embeddings [CLS] do BERT em batches."""
    dataset = ReviewDataset(texts, tokenizer, labels=labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    embeddings = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls)
            if (i + 1) % 20 == 0:
                print(f"    {i + 1}/{len(loader)} batches processados", end="\r")

    print()
    return np.vstack(embeddings)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de inferência: {device}\n")

    df_val, X_val, y_val = load_split("val")
    df_test, X_test, y_test = load_split("test")

    # Juntando validação e teste
    df_combined = pd.concat([df_val, df_test]).reset_index(drop=True)
    X_combined = pd.concat([X_val, X_test]).reset_index(drop=True)
    y_combined = pd.concat([y_val, y_test]).reset_index(drop=True)
    
    print(f"  Amostras carregadas : {len(df_combined):,}\n")

    tokenizer = load_tokenizer()
    
    model_bert = AutoModel.from_pretrained(MODEL_NAME).to(device)
    for param in model_bert.parameters():
        param.requires_grad = False
    
    lr_model_path = CHECKPOINT_DIR / "bertimbau_lr_model.pkl"
    with open(lr_model_path, "rb") as f:
        lr_model = pickle.load(f)

    X_embeddings = extract_cls_embeddings(X_combined, y_combined, tokenizer, model_bert, device)

    y_pred = lr_model.predict(X_embeddings)
    y_proba = lr_model.predict_proba(X_embeddings)[:, 1]

    f1 = f1_score(y_combined, y_pred, average="macro")
    auc = roc_auc_score(y_combined, y_proba)
    report_dict = classification_report(y_combined, y_pred, target_names=["negative", "positive"], output_dict=True)

    print(f"  F1-macro : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")

    # Criando os novos atributos diretamente na cópia do dataset original
    df_combined["inferencia_bertimbau"] = y_pred
    df_combined["probabilidade_bertimbau"] = np.round(y_proba, 4)
    
    # Exportando o dataset enriquecido
    csv_path = OUTPUT_DIR / "B2W-Reviews01_inferred_bertimbau.csv"
    df_combined.to_csv(csv_path, index=False)
    
    # Exportando métricas
    metrics = {
        "dataset_size": len(df_combined),
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
    json_path = OUTPUT_DIR / "metrics_inferred_bertimbau.json"
    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"  Dataset salvo em: {csv_path}")
    print(f"  Métricas salvas em: {json_path}")
    print("\nConcluído.")


if __name__ == "__main__":
    main()