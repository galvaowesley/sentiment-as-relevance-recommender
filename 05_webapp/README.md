# 05_webapp — SentiShop Demo

Aplicação web que demonstra o sistema de recomendação orientado por sentimento em uma interface que simula um site de e-commerce brasileiro. Integra os dois pipelines do projeto através de uma API REST.

## Arquitetura

```
05_webapp/
├── backend/          # FastAPI — wraps o ProductRecommender de 04_recommender/
│   ├── app.py        # Endpoints: /health /categories /products /recommend
│   ├── run.sh        # Launcher (configura PYTHONPATH automaticamente)
│   └── pyproject.toml
└── frontend/         # React + Vite SPA
    ├── src/
    │   ├── api.js                    # Fetch helpers para o backend
    │   ├── App.jsx                   # Rotas: / | /browse | /product/:id
    │   ├── styles.css
    │   └── components/
    │       ├── pages/
    │       │   ├── Home.jsx          # Grid de categorias
    │       │   ├── Browse.jsx        # Listagem paginada com sidebar e busca
    │       │   └── ProductDetail.jsx # Detalhe do produto + painel de recomendações
    │       ├── Header.jsx
    │       ├── CategorySidebar.jsx
    │       ├── ProductCard.jsx
    │       ├── SentimentBadge.jsx    # Badge colorido (verde/amarelo/vermelho)
    │       ├── Pagination.jsx
    │       ├── RecommendationsPanel.jsx
    │       └── CategoryIcon.jsx      # Ícones SVG para as 44 categorias B2W
    └── package.json
```

## Pré-requisitos

- Python ≥ 3.11 com dependências do projeto instaladas (`pip install -r requirements.txt` na raiz)
- Artefatos do recomendador já construídos (`04_recommender/artifacts/` populado — veja `04_recommender/README.md`)
- Node.js ≥ 18

## Como executar

### 1. Backend (Terminal 1)

```bash
cd 05_webapp/backend
bash run.sh
# API disponível em http://localhost:8000
# Documentação automática em http://localhost:8000/docs
```

O `run.sh` configura o `PYTHONPATH` automaticamente para incluir `04_recommender/`. Na primeira execução, o recomendador carrega os índices vetoriais do disco (alguns segundos).

### 2. Frontend (Terminal 2)

```bash
cd 05_webapp/frontend
npm install       # apenas na primeira vez
npm run dev
# Abrir http://localhost:5173
```

## Endpoints da API

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/categories` | Lista ordenada das categorias do catálogo |
| `GET` | `/products` | Catálogo paginado; suporta `?category=`, `?q=`, `?limit=`, `?offset=` |
| `GET` | `/products/{id}` | Produto individual |
| `GET` | `/recommend` | Recomendações re-ranqueadas por sentimento; suporta `?product_id=`, `?q=`, `?top_n=`, `?alpha=` |

## Fluxo de dados

```
Usuário → busca/navegação no frontend
  → GET /products?category=X ou /recommend?product_id=Y
  → ProductRecommender.load() (04_recommender/)
      → Qwen3-Embedding-0.6B vetoriza a query
      → HNSW ANN retrieval no corpus (5 000 produtos val+test sem neutros)
      → Re-rank: score = α · similaridade + (1−α) · sentiment_score
  → JSON → frontend renderiza cards com SentimentBadge
```

## Dados

| Conjunto | Origem | Papel |
|---|---|---|
| **Storefront** (5 000 produtos) | `B2W-Reviews01_train.csv` | Catálogo de navegação |
| **Corpus** (5 000 produtos) | `_no_neutral_val.csv` + `_no_neutral_test.csv` | Pool de recomendação com scores de sentimento mais nítidos (sem avaliações neutras) |
