# 04_recommender — Motor de Recomendação (Pipeline 2)

Sistema de recomendação de produtos por **busca semântica + re-ranqueamento por
sentimento**, preparado para ser consumido por um webapp que simula a página de um
produto de varejo.

Dado o produto de uma página (consulta `q`), o motor recupera os títulos mais
similares do corpus por busca vetorial (ANN) e os re-ordena combinando similaridade
semântica com o score de sentimento `S(p)` de cada produto.

## Arquitetura

| Módulo | Responsabilidade |
|---|---|
| `vectorizer/` | Embeddings dos títulos com **Qwen3-Embedding-0.6B** (1024-d, L2-norm) |
| `vector_store/` | Indexação e busca ANN por cosseno com **zvec** (HNSW) |
| `reranking/` | `S(p)` (sentimento) + re-rank ponderado `α·sim + (1−α)·S(p)` |
| `service/` | API FastAPI consumida pelo webapp |
| `recommender.py` | Fachada `ProductRecommender` (API pública do motor) |
| `build_index.py` | Constrói os artefatos (corpus + storefront) |

## Scores

```text
S(p)           = |avaliações positivas de p| / |total de avaliações de p|
Score_final(p) = α · sim(q, p) + (1 − α) · S(p)
```

`S(p)` é **mockado** (proporção de avaliações com `overall_rating ≥ 4`) até o
classificador de sentimento do Pipeline 1 ficar pronto — o classificador real
implementa o mesmo protocolo `SentimentScorer` e entra sem alterar o resto do motor.

## Dados

- **Corpus** (produtos recomendáveis): `no_neutral` **val + test**.
- **Storefront** (páginas navegáveis que geram as consultas): split **train** (com neutros).
- Produtos são deduplicados por `product_id`; recomendações excluem o próprio produto.

## Uso

Sempre a partir desta pasta, com o ambiente mamba `nlp_project`:

```bash
cd 04_recommender

# 1. Construir os artefatos (corpus de val+test, storefront de train)
mamba run -n nlp_project python build_index.py            # build completo
mamba run -n nlp_project python build_index.py --limit 2000   # build rápido

# 2. Demo de ponta a ponta no terminal
mamba run -n nlp_project python demo.py

# 3. Subir a API para o webapp
mamba run -n nlp_project uvicorn service.app:app --app-dir . --port 8000
#   GET /products            -> lista produtos da vitrine
#   GET /products/{id}       -> página de um produto
#   GET /recommend?product_id=<id>&top_n=5&alpha=0.7
```

Configuração central em `config.py` (modelo, `α`, `top_n`, `min_reviews`, device…).
O device é auto-detectado (CUDA → MPS → CPU) e pode ser forçado via
`RECOMMENDER_DEVICE`, tornando o código portável entre máquinas.

## Testes

```bash
cd 04_recommender
mamba run -n nlp_project pytest                 # testes rápidos (sem baixar modelo)
mamba run -n nlp_project pytest -m integration  # carrega o Qwen3-0.6B de verdade
```

Os testes rápidos usam um `FakeEmbedder` determinístico e o backend zvec real,
cobrindo retrieval, re-rank, persistência e os contratos HTTP sem baixar pesos.
