# reranking — Re-ranqueamento por Sentimento

Re-ordena os K candidatos retornados pelo banco vetorial combinando similaridade semântica e score de sentimento:

```
Score_final(p) = α · sim(q, p) + (1 − α) · S(p)
```

- `sim(q, p)`: similaridade de cosseno entre os embeddings do produto de consulta e do candidato
- `S(p)`: proporção de avaliações positivas do produto (calculado em `../vectorizer/`)
- `α`: hiperparâmetro que controla o trade-off entre relevância semântica e positividade percebida

Os produtos são ordenados de forma decrescente por `Score_final` e os top-k são retornados como recomendações finais.
