"""
Evaluate LLM sentiment predictions (val+test) produced by infer_llm.py.

Steps:
    1. Load the predictions CSV, drop rows that failed during inference
    2. Evaluate val, test, and val+test combined (F1-macro, AUC-ROC, Precisão,
       Revocação, F1 por classe, Matriz de Confusão — mesma tabela impressa
       pelos outros classificadores do projeto)
    3. Aggregate token usage / response time per split and overall
    4. Save results in 03_sentiment_classifier/checkpoints/ using the same
       JSON shape as tfidf_bigrams_lr_results.json / bertimbau_lr_results.json,
       plus an extra "overall" block for val+test combined

Usage:
    python 03_sentiment_classifier/llm/evaluate_llm.py --predictions 03_sentiment_classifier/checkpoints/llm_qwen_qwen3-4b-2507_predictions.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT / "03_sentiment_classifier" / "checkpoints"


def load_predictions(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(path)
    df["llm_error"] = df["llm_error"].fillna("")
    total_by_split = df["split"].value_counts().to_dict()

    ok = df[df["llm_error"] == ""].copy()
    ok["true_polarity"] = ok["true_polarity"].astype(int)
    ok["llm_predicted_polarity"] = ok["llm_predicted_polarity"].astype(int)

    for split_name, total in total_by_split.items():
        n_ok = int((ok["split"] == split_name).sum())
        n_failed = total - n_ok
        pct = 100 * n_failed / total if total else 0.0
        print(f"  {split_name}: {n_ok:,}/{total:,} sucesso ({pct:.1f}% falharam)")

    return ok, total_by_split


def evaluate_split(y_true, y_pred) -> dict:
    f1 = f1_score(y_true, y_pred, average="macro")
    # Sem probabilidade calibrada disponível (classificação one-shot só retorna
    # o rótulo) — AUC-ROC é calculado a partir da predição binária dura.
    auc = roc_auc_score(y_true, y_pred)

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
            },
        },
    }


def performance_stats(df: pd.DataFrame) -> dict:
    return {
        "avg_input_tokens": round(df["llm_input_tokens"].mean(), 2),
        "avg_output_tokens": round(df["llm_output_tokens"].mean(), 2),
        "avg_response_time_s": round(df["llm_response_time_s"].mean(), 4),
        "total_response_time_s": round(df["llm_response_time_s"].sum(), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM sentiment predictions and save a results JSON.")
    parser.add_argument("--predictions", type=Path, required=True, help="CSV de predições gerado por infer_llm.py.")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Identificador cru do modelo no LM Studio, para registro no JSON (ex.: 'qwen/qwen3-4b-2507'). "
             "Default: nome derivado do arquivo de predições.",
    )
    parser.add_argument("--results-output", type=Path, default=None, help="Caminho do JSON de resultados.")
    args = parser.parse_args()

    model_name = args.predictions.stem.removesuffix("_predictions")
    llm_model_identifier = args.model or model_name
    results_output = args.results_output or CHECKPOINT_DIR / f"{model_name}_results.json"

    print("=" * 60)
    print("STEP 1 — Loading predictions")
    print("=" * 60)
    ok, total_by_split = load_predictions(args.predictions)

    val = ok[ok["split"] == "val"]
    test = ok[ok["split"] == "test"]

    print("\n" + "=" * 60)
    print("STEP 2a — Evaluation on validation set")
    print("=" * 60)
    val_metrics = evaluate_split(val["true_polarity"], val["llm_predicted_polarity"])
    val_metrics.update(performance_stats(val))

    print("\n" + "=" * 60)
    print("STEP 2b — Evaluation on test set")
    print("=" * 60)
    test_metrics = evaluate_split(test["true_polarity"], test["llm_predicted_polarity"])
    test_metrics.update(performance_stats(test))

    print("\n" + "=" * 60)
    print("STEP 2c — Evaluation overall (val+test combined)")
    print("=" * 60)
    overall_metrics = evaluate_split(ok["true_polarity"], ok["llm_predicted_polarity"])
    overall_metrics.update(performance_stats(ok))

    print("\n" + "=" * 60)
    print("STEP 3 — Saving results")
    print("=" * 60)
    total_size = sum(total_by_split.values())
    results = {
        "model": model_name,
        "llm_model_identifier": llm_model_identifier,
        "prompt_config": "prompt_one_shot.yaml",
        "val": val_metrics,
        "test": test_metrics,
        "overall": overall_metrics,
        "val_size": total_by_split.get("val", 0),
        "test_size": total_by_split.get("test", 0),
        "overall_size": total_size,
        "val_failed": total_by_split.get("val", 0) - len(val),
        "test_failed": total_by_split.get("test", 0) - len(test),
        "overall_failed": total_size - len(ok),
    }
    results_output.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"  results.json → {results_output}")
    print("\nDone.")


if __name__ == "__main__":
    main()
