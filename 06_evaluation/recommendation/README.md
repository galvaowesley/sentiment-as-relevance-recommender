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
