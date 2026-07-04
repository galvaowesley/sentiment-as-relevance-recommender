# service — API FastAPI do Recomendador (standalone)

Versão enxuta da API que expõe o `ProductRecommender` sobre HTTP, para uso e testes isolados do Pipeline 2. O webapp de produção (`05_webapp/backend/`) é um superconjunto desta API (adiciona categorias, listagem de avaliações, enriquecimento de produtos e serve o SPA).

## Endpoints (GET)

`/health` · `/products` (`limit`, `offset`, `category`, `q`) · `/products/{product_id}` · `/recommend` (`product_id`, `q`, `top_n`, `alpha`)

O índice é carregado de forma preguiçosa (singleton) e aquecido no startup, lendo `04_recommender/artifacts/`.

## Execução

```bash
cd 04_recommender
uvicorn service.app:app --app-dir . --port 8000
```
