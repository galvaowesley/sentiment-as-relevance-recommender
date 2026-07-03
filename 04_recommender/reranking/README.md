# reranking — Re-ranqueamento por Sentimento

Re-ordena os K candidatos retornados pelo banco vetorial combinando similaridade
semântica e score de sentimento:

```text
Score_final(p) = α · sim(q, p) + (1 − α) · S(p)
```

- `sim(q, p)`: similaridade de cosseno entre os embeddings do produto de consulta e do candidato
- `S(p)`: proporção de avaliações positivas do produto, em `[0, 1]`
- `α`: hiperparâmetro que controla o trade-off entre relevância semântica e positividade percebida

`S(p)` é calculado por um `SentimentScorer` (`sentiment.py`). Enquanto o
classificador do Pipeline 1 não está pronto, usa-se o `MockSentimentScorer`
(proporção de avaliações com `overall_rating ≥ 4`), computado na construção do
índice (`build_index.py`). O re-rank em si (`reranker.py`) é uma função pura.

Os produtos são ordenados de forma decrescente por `Score_final` e os top-k são
retornados como recomendações finais.
