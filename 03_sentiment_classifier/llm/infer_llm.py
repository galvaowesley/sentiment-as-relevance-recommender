"""
Run one-shot sentiment inference over val+test using a local LLM (LM Studio).

Steps:
    1. Load no-neutral val + test splits (never used for training)
    2. Light per-field cleaning (fillna + strip — no aggressive normalization,
       case/punctuation matter for an LLM reading sentiment)
    3. One independent request per review — fresh system+user messages each
       time, no conversation object reused across rows, structured JSON output
    4. Append each result to a predictions CSV as soon as it's ready (resumable)

Usage:
    python 03_sentiment_classifier/llm/infer_llm.py --model "qwen/qwen3-4b-2507" [--limit N] [--output PATH]
"""

import argparse
import sys
import time
from pathlib import Path

import lmstudio as lms
import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_preprocessing"))

from common.preprocess import map_polarity  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "03_sentiment_classifier" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_CONFIG_PATH = Path(__file__).resolve().parent / "prompt_one_shot.yaml"

SENTIMENT_TO_POLARITY = {"negative": 0, "positive": 1}

ORIGINAL_COLUMNS = [
    "submission_date", "reviewer_id", "product_id", "product_name", "product_brand",
    "site_category_lv1", "site_category_lv2", "review_title", "overall_rating",
    "recommend_to_a_friend", "review_text", "reviewer_birth_year", "reviewer_gender",
    "reviewer_state",
]
OUTPUT_COLUMNS = ORIGINAL_COLUMNS + [
    "row_id", "split", "true_polarity", "llm_raw_response", "llm_predicted_polarity",
    "llm_error", "llm_input_tokens", "llm_output_tokens", "llm_response_time_s",
]


def load_prompt_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Light per-field cleaning for LLM prompting: fillna + strip only.

    Deliberately does not reuse common.preprocess.normalize_text — lowercasing
    and punctuation stripping there are tuned for TF-IDF/BERT, not for an LLM
    that can read case/punctuation as sentiment signal directly.
    """
    df = df.copy()
    df["review_title"] = df["review_title"].fillna("").astype(str).str.strip()
    df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
    return df


def load_split(name: str, limit: int | None) -> pd.DataFrame:
    path = DATA_DIR / f"B2W-Reviews01_no_neutral_{name}.csv"
    df = pd.read_csv(path)
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)
    df = clean_text_columns(df)

    polarity = map_polarity(df)
    assert polarity.notna().all(), f"{name}: found unexpected neutral (rating 3) rows"
    df["true_polarity"] = polarity.astype(int)
    df["split"] = name
    df["row_id"] = [f"{name}:{i}" for i in range(len(df))]
    return df


def load_concat_val_test(limit: int | None) -> pd.DataFrame:
    val = load_split("val", limit)
    test = load_split("test", limit)
    return pd.concat([val, test], ignore_index=True)


def classify_row(
    model,
    system_prompt: str,
    user_template: str,
    response_schema: dict,
    title: str,
    text: str,
) -> dict:
    """One independent, stateless request: fresh system+user messages built
    from scratch on every call, nothing carried over between rows."""
    user_message = user_template.format(title=title, text=text)
    start = time.perf_counter()
    try:
        result = model.respond(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
            },
            response_format=response_schema,
        )
        elapsed = time.perf_counter() - start

        sentiment = result.parsed["sentiment"]
        if sentiment not in SENTIMENT_TO_POLARITY:
            raise ValueError(f"malformed_response: unexpected sentiment {sentiment!r}")

        stats = result.stats
        return {
            "sentiment": sentiment,
            "error": "",
            "input_tokens": getattr(stats, "prompt_tokens_count", None),
            "output_tokens": getattr(stats, "predicted_tokens_count", None),
            "response_time_s": round(elapsed, 4),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "sentiment": "",
            "error": f"{type(exc).__name__}: {exc}",
            "input_tokens": None,
            "output_tokens": None,
            "response_time_s": round(elapsed, 4),
        }


def prepare_output(output_path: Path) -> set[str]:
    """Return the row_ids already resolved successfully in a prior run.

    Any previously-failed row is dropped from the file (compaction) so it
    gets naturally retried on this run instead of accumulating duplicates.
    """
    if not output_path.exists():
        return set()

    existing = pd.read_csv(output_path, dtype={"row_id": str})
    existing["llm_error"] = existing["llm_error"].fillna("")
    ok = existing[existing["llm_error"] == ""]

    n_dropped = len(existing) - len(ok)
    if n_dropped:
        print(f"  Retomando: descartando {n_dropped} linha(s) com erro anterior para nova tentativa.")
    ok.to_csv(output_path, index=False)
    return set(ok["row_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-shot LLM sentiment inference over val+test.")
    parser.add_argument(
        "--model", required=True,
        help="Identificador do modelo carregado no LM Studio (ex.: 'qwen/qwen3-4b-2507'). "
             "Descubra os disponíveis com `lms ls`.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limita cada split (val e test) às N primeiras linhas — útil para smoke test.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Caminho do CSV de predições.")
    parser.add_argument("--prompt-config", type=Path, default=PROMPT_CONFIG_PATH, help="Caminho do YAML de prompt.")
    args = parser.parse_args()

    output_path = args.output or CHECKPOINT_DIR / f"llm_{sanitize_model_id(args.model)}_predictions.csv"

    print("=" * 60)
    print("STEP 1 — Loading prompt config")
    print("=" * 60)
    prompt_cfg = load_prompt_config(args.prompt_config)
    system_prompt = prompt_cfg["system_prompt"]
    user_template = prompt_cfg["user_template"]
    response_schema = prompt_cfg["response_schema"]
    print(f"  Config: {args.prompt_config}")

    print("\n" + "=" * 60)
    print("STEP 2 — Loading data (val + test, no neutral)")
    print("=" * 60)
    df = load_concat_val_test(args.limit)
    print(f"  Val + Test: {len(df):,} samples")

    print("\n" + "=" * 60)
    print(f"STEP 3 — Connecting to LM Studio model: {args.model}")
    print("=" * 60)
    model = lms.llm(args.model)
    print("  Connected.")

    print("\n" + "=" * 60)
    print("STEP 4 — Resuming from existing output (if any)")
    print("=" * 60)
    done_ids = prepare_output(output_path)
    pending = df[~df["row_id"].isin(done_ids)].reset_index(drop=True)
    print(f"  Já concluídas: {len(done_ids):,} | Pendentes: {len(pending):,}")

    print("\n" + "=" * 60)
    print("STEP 5 — Running inference (one independent request per review)")
    print("=" * 60)
    write_header = not (output_path.exists() and output_path.stat().st_size > 0)

    n_ok, n_err = 0, 0
    for row in tqdm(pending.itertuples(index=False), total=len(pending), desc="Classificando"):
        result = classify_row(model, system_prompt, user_template, response_schema, row.review_title, row.review_text)
        sentiment = result["sentiment"]

        out_row = {col: getattr(row, col) for col in ORIGINAL_COLUMNS}
        out_row.update({
            "row_id": row.row_id,
            "split": row.split,
            "true_polarity": row.true_polarity,
            "llm_raw_response": sentiment,
            "llm_predicted_polarity": SENTIMENT_TO_POLARITY.get(sentiment, ""),
            "llm_error": result["error"],
            "llm_input_tokens": result["input_tokens"],
            "llm_output_tokens": result["output_tokens"],
            "llm_response_time_s": result["response_time_s"],
        })
        pd.DataFrame([out_row], columns=OUTPUT_COLUMNS).to_csv(output_path, mode="a", header=write_header, index=False)
        write_header = False

        if result["error"]:
            n_err += 1
        else:
            n_ok += 1

    print(f"\n  Sucesso: {n_ok:,} | Erro: {n_err:,}")
    print(f"  predictions.csv → {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
