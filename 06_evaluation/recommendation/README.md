# recommendation — Avaliação do Pipeline 2

Métricas de ranking para o módulo de recomendação re-ranqueada por sentimento.

## Métricas

| Métrica | Descrição |
|---|---|
| Precision@k | Proporção de itens relevantes nos k primeiros retornados |
| Recall@k | Proporção de itens relevantes recuperados dentre todos os relevantes |
| NDCG@k | Penaliza a posição dos itens relevantes — favorece listas com os melhores itens no topo |

## Critério de relevância

- **Altamente relevante:** S(p) > 0.7
- **Pouco relevante:** S(p) < 0.3

## Amostra de avaliação — `recommendation_eval.json`

Recomendações geradas pelo motor (`04_recommender`) com `top_n=10`, `alpha=0.7`
(`Score = α·sim(q,p) + (1−α)·S(p)`) e `recommend_min_reviews=5`. A amostra
contrasta **duas categorias** — uma de baixo risco e uma de alto risco de
recomendação ruim — com **2 produtos de consulta cada** e métricas de qualidade
por consulta (`mean_similarity`, `mean_sentiment`, `n_low_relevance` com S<0.3,
`n_off_topic` com sim<0.5, `n_same_category`).

### Racional das categorias

| Bucket | Categoria | Por quê |
|---|---|---|
| `low_risk_high_volume` | **Celulares e Smartphones** | Categoria grande e muito avaliada (1247 produtos, 8363 avaliações, 250 recomendáveis). Produtos coerentes e bem avaliados → recomenda smartphones similares com S(p) alto e alta similaridade. |
| `high_risk` | **Móveis** | Categoria com **mais produtos** do catálogo (2028) porém **pior sentimento** entre as grandes (S médio dos recomendáveis = 0.39; 23/61 recomendáveis com S<0.3) → alto risco de recomendação ruim. |

Os produtos de consulta são recomendáveis (`num_reviews ≥ 5`); os de Móveis foram
escolhidos por expor os dois modos de falha do sistema:

- **`13628860` Cama Solteiro Barcelona** (S=0.17) — **deriva de tema**: com
  similaridade média ~0.44, as 10 recomendações ficam fora do tema (ex.: #1 =
  *Bomba Tira-Leite Materno*, S=1.0). O re-rank por sentimento puxa itens de S
  alto de outras categorias, sacrificando a relevância semântica.
- **`24266805` Guarda Roupa Casal Conjugado** (S=0.22) — **falha
  relevância-sentimento**: a recomendação #1 é um guarda-roupa quase idêntico com
  **S=0.20** (produto malavaliado), promovido ao topo porque a similaridade
  altíssima (~0.86) domina o re-rank.

### Leitura dos resultados

Com **α=0.7** o re-rank por sentimento é robusto: raramente recomenda produtos de
S(p) baixo, pois favorece vizinhos de sentimento alto. Assim, o "risco de
recomendação ruim" manifesta-se sobretudo como **item fora do tema** (baixa
similaridade), não como sentimento baixo — exceto em **quase-duplicatas**, em que
a similaridade altíssima promove ao topo um item de S(p) baixo apesar do re-rank.

> Fonte: JSON gerado a partir da API do motor (`04_recommender`), sobre o corpus
> inferido pelo BERTimbau (val+test).
