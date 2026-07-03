"""
Script for random sampling of unique items from a dataset.

Steps:
    1. Load the target dataset (e.g., the output from the inference script).
    2. Remove neutral ratings (e.g., rating == 3).
    3. Drop duplicate entries based on a specific column to ensure uniqueness.
    4. Randomly sample N items without replacement.
    5. Save the sampled dataset locally in CSV and JSON formats.

Usage:
    python sample_data.py --input B2W-Reviews01_inferred_tfidf_both.csv --n_samples 500 --seed 42
"""

import argparse
from pathlib import Path

import pandas as pd

# Configuração de caminhos baseada no script original
ROOT = Path(__file__).resolve().parents[1]

# Assumindo que vamos ler da pasta de outputs de inferência
INPUT_DIR = ROOT / "03_sentiment_classifier" / "inference" / "outputs"
OUTPUT_DIR = ROOT / "03_sentiment_classifier" / "inference" / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seleciona amostras aleatórias e únicas de um dataset, excluindo avaliações neutras.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Nome do arquivo CSV de entrada (ex: B2W-Reviews01_inferred_tfidf_both.csv)."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Número de amostras aleatórias para extrair."
    )
    parser.add_argument(
        "--unique_col",
        type=str,
        default="review_text",
        help="Coluna usada como referência para remover duplicatas e garantir itens únicos."
    )
    parser.add_argument(
        "--rating_col",
        type=str,
        default="overall_rating",
        help="Nome da coluna de avaliação para aplicar o filtro de exclusão da nota 3."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória para reprodutibilidade."
    )
    args = parser.parse_args()

    input_path = INPUT_DIR / args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

    print(f"Carregando dataset: {input_path.name}...")
    df = pd.read_csv(input_path)
    print(f"  Tamanho original: {len(df):,}")

    # 1. Excluir notas 3 (avaliações neutras)
    if args.rating_col in df.columns:
        df = df[df[args.rating_col] != 3].reset_index(drop=True)
        print(f"  Tamanho após remover nota 3 (coluna '{args.rating_col}'): {len(df):,}")
    else:
        print(f"  Aviso: Coluna de notas '{args.rating_col}' não encontrada. Nenhuma nota foi excluída.")

    if args.unique_col in df.columns:
        df_unique = df.drop_duplicates(subset=[args.unique_col]).reset_index(drop=True)
        print(f"  Tamanho após remover duplicatas na coluna '{args.unique_col}': {len(df_unique):,}")
    else:
        print(f"  Aviso: Coluna '{args.unique_col}' não encontrada. Removendo duplicatas exatas de todas as colunas.")
        df_unique = df.drop_duplicates().reset_index(drop=True)
        print(f"  Tamanho após remover linhas inteiras duplicadas: {len(df_unique):,}")

    n_samples = min(args.n_samples, len(df_unique))
    if n_samples < args.n_samples:
        print(f"  Aviso: O número solicitado ({args.n_samples}) é maior que o dataset disponível. Extraindo todos os {n_samples} itens.")

    print(f"Extraindo {n_samples:,} amostras aleatórias (Seed: {args.seed})...")
    df_sampled = df_unique.sample(n=n_samples, replace=False, random_state=args.seed).reset_index(drop=True)

    output_base_name = f"sampled_{n_samples}_{input_path.stem}"
    csv_out = OUTPUT_DIR / f"{output_base_name}.csv"
    json_out = OUTPUT_DIR / f"{output_base_name}.json"

    df_sampled.to_csv(csv_out, index=False)
    
    df_sampled.to_json(json_out, orient="records", force_ascii=False, indent=2)

    print("\nConcluído. Arquivos salvos em:")
    print(f"  -> CSV : {csv_out}")
    print(f"  -> JSON: {json_out}")


if __name__ == "__main__":
    main()