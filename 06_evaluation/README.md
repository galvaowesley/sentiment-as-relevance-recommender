# 06_evaluation — Avaliação

Reúne os resultados quantitativos e qualitativos dos dois pipelines do projeto.

## Conteúdo

| Pasta | Escopo |
|---|---|
| `classification/` | Métricas do Pipeline 1 (F1-macro, AUC-ROC, matriz de confusão) + amostra qualitativa |
| `recommendation/` | Métricas do Pipeline 2 (NDCG@k, Precision@k, Recall@k) via `objective_metrics.py` |

> A avaliação qualitativa (inspeção manual de predições) é feita a partir de amostras extraídas com `03_sentiment_classifier/sample_data.py`; a amostra do modelo campeão está em `classification/` (`sampled_50_*`).
