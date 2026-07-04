# results_from_best_model — Inferência do Melhor Modelo (Pipeline 1)

Saída do **melhor classificador de sentimento** do Pipeline 1 (**BERTimbau**) aplicada
aos conjuntos de **validação + teste** do B2W-Reviews01. Cada linha é uma avaliação com a
polaridade prevista pelo modelo.

Este é o dataset que **alimenta o motor de recomendação** (`04_recommender/`): a partir
dele calcula-se o score de positividade por produto `S(p)` e constroem-se o corpus e a
vitrine vetorizados. É a ponte entre o Pipeline 1 (classificação) e o Pipeline 2
(recomendação).

## Arquivo

| Arquivo | Descrição |
|---|---|
| `B2W-Reviews01_inferred_bertimbau.csv` | Avaliações de val + test com a predição do BERTimbau por review. |

## Colunas

Traz todas as colunas originais do B2W-Reviews01 (`product_id`, `product_name`,
`product_brand`, `site_category_lv1/lv2`, `overall_rating`, `review_text`, …) e acrescenta:

| Coluna | Descrição |
|---|---|
| `inferencia_bertimbau` | Rótulo predito por review: `1` = positivo, `0` = negativo. |
| `probabilidade_bertimbau` | Probabilidade da classe positiva, em `[0, 1]`. |

## Como é usado

`04_recommender/build_index.py` lê este arquivo (caminho definido em
`04_recommender/config.py` → `DEFAULT_INFERRED_CSV`) e:

- calcula `S(p) = |reviews com inferencia_bertimbau == 1| / |reviews do produto|`
  (via `PredictedLabelSentimentScorer`);
- grava a tabela por produto em `data/processed/sentiment_scores.parquet`;
- vetoriza **todos** os produtos (corpus e vitrine partem daqui, para coerência).

> Observação: os rótulos vêm do modelo (não das notas). A nota `overall_rating` continua
> disponível, mas o `S(p)` do recomendador é baseado em `inferencia_bertimbau`.
