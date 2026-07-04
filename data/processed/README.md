# processed — Dados Processados

Datasets derivados do corpus original gerados pelos scripts de `02_preprocessing/`.

## Arquivos esperados

| Arquivo | Descrição |
|---|---|
| `train.parquet` | 70% do corpus com polaridade mapeada (estratificado) |
| `val.parquet` | 15% para ajuste de hiperparâmetros |
| `test.parquet` | 15% para avaliação final (não tocar até o fim) |
| `sentiment_scores.parquet` | Score S(p) por produto, gerado por `04_recommender/build_index.py` a partir da inferência do BERTimbau. Colunas: `product_id, num_reviews, num_positive, sentiment_score` (`sentiment_score = num_positive / num_reviews`). |

> Mapeamento de polaridade: notas 1–2 → negativo, 4–5 → positivo, nota 3 → descartada.
