# processed — Dados Processados

Datasets derivados do corpus original, gerados pelos scripts de `02_preprocessing/` (`make splits`).

## Arquivos

| Arquivo | Descrição |
|---|---|
| `B2W-Reviews01_{train,val,test}.csv` | Splits com polaridade mapeada, **mantendo** avaliações neutras (usados na inferência) |
| `B2W-Reviews01_no_neutral_{train,val,test}.csv` | Splits **sem** neutros (usados no treino dos modelos) |
| `sentiment_scores.parquet` | Score `S(p)` por produto, gerado por `04_recommender/build_index.py`. Colunas: `product_id, num_reviews, num_positive, sentiment_score` (`sentiment_score = num_positive / num_reviews`) |

> Mapeamento de polaridade: notas 1–2 → negativo, 4–5 → positivo, nota 3 → descartada (nos splits `no_neutral`). Proporção padrão dos splits: 60/20/20 estratificado.
