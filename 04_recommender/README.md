# 04_recommender — Motor de Recomendação (Pipeline 2)

Constrói e avalia o motor de recomendação orientado por sentimento. Consome o modelo campeão do Pipeline 1 para calcular um score de positividade S(p) por produto, vetoriza os títulos de produtos com o Serafim PT* e re-ranqueia os candidatos recuperados por busca vetorial.

## Conteúdo

| Pasta | Responsabilidade |
|---|---|
| `vectorizer/` | Geração dos embeddings de títulos com Serafim PT* |
| `vector_store/` | Indexação e busca ANN por similaridade de cosseno |
| `reranking/` | Re-ranqueamento ponderado por similaridade + sentimento |

## Score de relevância por produto

```
S(p) = |avaliações positivas de p| / |total de avaliações de p|
```

Apenas produtos com ao menos 5 avaliações são incluídos no índice.

## Score final de recomendação

```
Score_final(p) = α · sim(q, p) + (1 − α) · S(p)
```

onde α ∈ [0, 1] é ajustado empiricamente durante a validação.
