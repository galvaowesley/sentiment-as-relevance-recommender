# data — Dados do Projeto

Armazena o corpus original, os datasets derivados, as inferências do modelo campeão e amostras para testes.

## Conteúdo

| Pasta | Descrição |
|---|---|
| `raw/` | Corpus B2W-Reviews01 original (não versionado no git) |
| `processed/` | Splits treino/val/teste (`.csv`) e `sentiment_scores.parquet` |
| `results_from_best_model/` | CSV inferido pelo modelo campeão (BERTimbau), consumido pelo recomendador |
| `samples/` | Amostras pequenas para testes rápidos (planejado) |

> O corpus bruto e os embeddings (`.npy`) estão listados no `.gitignore`. Consulte `raw/README.md` para instruções de download.
