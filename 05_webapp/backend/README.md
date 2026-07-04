# backend — API REST (FastAPI)

Serviço FastAPI da demo **SentiShop**. Embrulha o `ProductRecommender` de `04_recommender/` (injetado via `PYTHONPATH`) e expõe o storefront simulado, as recomendações e os rótulos de sentimento por avaliação. Em produção, também serve o SPA compilado (`../frontend/dist`) na raiz `/`.

Não carrega nenhum modelo de sentimento em tempo de execução: o score `S(p)` já vem pré-computado nos catálogos do índice, e os rótulos por review vêm do CSV inferido pelo campeão.

## Fontes de dados carregadas

- Índice do recomendador: `04_recommender/artifacts/` (via `ProductRecommender.load()`)
- Avaliações por produto: `03_sentiment_classifier/inference/outputs/B2W-Reviews01_inferred_bertimbau.csv` (fallback: `data/processed/B2W-Reviews01_{val,test}.csv`), colunas `inferencia_bertimbau` e `probabilidade_bertimbau`

## Endpoints (todos GET)

| Rota | Descrição |
|---|---|
| `/health` | Liveness (`{"status":"ok"}`) — usado pelo healthcheck do Railway |
| `/categories` | Categorias nível 1 |
| `/categories/lv2?cat1=` | Subcategorias nível 2 |
| `/products/count` | Contagem de produtos (com filtros) |
| `/products?limit=&offset=&category=&cat2=&q=` | Lista paginada do storefront |
| `/products/{product_id}` | Detalhe do produto (enriquecido com `avg_rating`, `review_count`) |
| `/products/{product_id}/reviews` | Avaliações com rótulo de sentimento e confiança |
| `/recommend?product_id=&q=&top_n=&alpha=` | Recomendações re-ranqueadas (`Score = α·sim + (1−α)·S(p)`) |

Docs interativas em `/docs` (Swagger, geradas automaticamente pelo FastAPI).

## Execução

```bash
# via Makefile na raiz
make backend            # dev, :8000 com --reload
make webapp             # compila o frontend e serve tudo na porta única :8000

# ou diretamente
bash 05_webapp/backend/run.sh   # PYTHONPATH=04_recommender uvicorn app:app --port 8000 --reload
```

Stack: FastAPI + Uvicorn (`pyproject.toml` = `sentishop-backend`). As dependências estão no `requirements.txt` da raiz.
