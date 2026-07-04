# reranking — Re-ranqueamento por Sentimento

Re-ordena os K candidatos retornados pelo banco vetorial combinando similaridade
semântica e score de sentimento:

```text
Score_final(p) = α · sim(q, p) + (1 − α) · S(p)
```

- `sim(q, p)`: similaridade de cosseno entre os embeddings do produto de consulta e do candidato
- `S(p)`: proporção de avaliações positivas do produto, em `[0, 1]`
- `α`: hiperparâmetro que controla o trade-off entre relevância semântica e positividade percebida

`S(p)` é calculado por um `SentimentScorer` (`sentiment.py`) na construção do índice
(`build_index.py`). Corpus e vitrine vêm do mesmo CSV inferido, então ambos usam o
`PredictedLabelSentimentScorer`: proporção de avaliações classificadas como positivas
pelo Pipeline 1 (`inferencia_bertimbau == 1`). O `MockSentimentScorer` (proporção de
`overall_rating ≥ 4`) fica apenas para os testes.

O re-rank em si (`reranker.py`) é uma função pura. Só entram no re-rank produtos com
`num_reviews ≥ recommend_min_reviews` (filtrados em query-time); os produtos elegíveis
são ordenados de forma decrescente por `Score_final` e os top-k viram as recomendações.
