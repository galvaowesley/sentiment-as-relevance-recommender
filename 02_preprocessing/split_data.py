"""
Split a CSV dataset into train, validation, and test sets.

Usage:
    python split_data.py --input <path> [options]

Examples:
    python split_data.py --input data/raw/B2W-Reviews01.csv
    python split_data.py --input data/raw/B2W-Reviews01.csv --train 0.7 --val 0.15 --test 0.15
    python split_data.py --input data/raw/B2W-Reviews01.csv --output-dir data/processed --seed 123
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    stratify_col: str | None = None,
    seed: int = 42,
) -> dict[str, Path]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.4f}"
        )

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"  Total rows: {len(df):,}")

    stratify = df[stratify_col] if stratify_col else None

    # First split: train vs (val + test)
    df_train, df_temp = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=stratify,
    )

    # Second split: val vs test from the remainder
    relative_val = val_ratio / (val_ratio + test_ratio)
    stratify_temp = df_temp[stratify_col] if stratify_col else None
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - relative_val),
        random_state=seed,
        stratify=stratify_temp,
    )

    stem = input_path.stem
    splits = {
        "train": output_dir / f"{stem}_train.csv",
        "val": output_dir / f"{stem}_val.csv",
        "test": output_dir / f"{stem}_test.csv",
    }

    df_train.to_csv(splits["train"], index=False)
    df_val.to_csv(splits["val"], index=False)
    df_test.to_csv(splits["test"], index=False)

    total = len(df)
    print(f"  train : {len(df_train):,} rows ({len(df_train)/total:.1%}) → {splits['train']}")
    print(f"  val   : {len(df_val):,} rows ({len(df_val)/total:.1%}) → {splits['val']}")
    print(f"  test  : {len(df_test):,} rows ({len(df_test)/total:.1%}) → {splits['test']}")

    return splits


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a CSV dataset into train / val / test sets."
    )
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files. Defaults to the same directory as --input.",
    )
    parser.add_argument(
        "--train", type=float, default=0.60, metavar="RATIO", help="Train ratio (default: 0.60)."
    )
    parser.add_argument(
        "--val", type=float, default=0.20, metavar="RATIO", help="Validation ratio (default: 0.20)."
    )
    parser.add_argument(
        "--test", type=float, default=0.20, metavar="RATIO", help="Test ratio (default: 0.20)."
    )
    parser.add_argument(
        "--stratify",
        default=None,
        metavar="COLUMN",
        help="Column name to use for stratified splitting.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    output_dir = args.output_dir or str(Path(args.input).parent)

    split_dataset(
        input_path=args.input,
        output_dir=output_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        stratify_col=args.stratify,
        seed=args.seed,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
