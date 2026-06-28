# 05_webapp — SentiShop Demo

Aplicação web que demonstra o sistema de recomendação orientado por sentimento em uma interface que simula um site de e-commerce brasileiro. Integra os dois pipelines do projeto através de uma API REST.

## Arquitetura

```
05_webapp/
├── backend/          # FastAPI — wraps ProductRecommender de 04_recommender/
│   ├── app.py        # Endpoints da API + carregamento de reviews (val+test)
│   ├── run.sh        # Launcher (configura PYTHONPATH automaticamente)
│   └── pyproject.toml
├── frontend/         # React + Vite SPA
│   ├── src/
│   │   ├── api.js                        # Fetch helpers para o backend
│   │   ├── App.jsx                       # Rotas: / | /product/:id
│   │   ├── styles.css
│   │   └── components/
│   │       ├── pages/
│   │       │   ├── Browse.jsx            # Grid de produtos (sidebar + filtros + paginação)
│   │       │   └── ProductDetail.jsx     # Detalhe + avaliações + painel de recomendações
│   │       ├── Header.jsx                # Busca por nome ou ID numérico
│   │       ├── CategorySidebar.jsx       # Sidebar colapsável lv1/lv2 (accordion)
│   │       ├── ProductCard.jsx           # Card com estrelas + avg_rating
│   │       ├── Pagination.jsx            # Paginação com número de páginas
│   │       ├── RecommendationsPanel.jsx  # Pipeline steps + score bars + fórmula
│   │       └── CategoryIcon.jsx          # Ícones SVG para as categorias B2W
│   └── package.json
└── handoff.md        # Metodologia completa, decisões e próximas iterações
```

## Pré-requisitos

- Python ≥ 3.11 com dependências instaladas (`pip install -r requirements.txt` na raiz)
- Artefatos do recomendador construídos (`04_recommender/artifacts/` populado — veja `04_recommender/README.md`)
- Node.js ≥ 18

## Como executar

### 1. Backend (Terminal 1)

```bash
cd 05_webapp/backend
bash run.sh
# API disponível em http://localhost:8000
# Documentação em http://localhost:8000/docs
```

Na primeira execução, o backend carrega os índices vetoriais do disco e os CSVs de val+test para enriquecimento de dados (avg_rating, reviews). Pode levar alguns segundos.

### 2. Frontend (Terminal 2)

```bash
cd 05_webapp/frontend
npm install       # apenas na primeira vez
npm run dev
# Abrir http://localhost:5173
```

## Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/categories` | Categorias lv1 do catálogo |
| `GET` | `/categories/lv2?cat1=X` | Subcategorias lv2 para um lv1 |
| `GET` | `/products` | Catálogo paginado; suporta `?category=`, `?cat2=`, `?q=`, `?limit=`, `?offset=` |
| `GET` | `/products/count` | Total de produtos com os mesmos filtros (para paginação) |
| `GET` | `/products/{id}` | Produto individual com avg_rating, review_count e "Sobre o produto" |
| `GET` | `/products/{id}/reviews` | Todas as avaliações do produto (val+test, incluindo rating=3) |
| `GET` | `/recommend` | Top-N recomendações re-ranqueadas; suporta `?product_id=`, `?q=`, `?top_n=`, `?alpha=` |

## Como funciona a similaridade

> **Os embeddings são pré-computados offline. A busca de vizinhos acontece em tempo real via HNSW (ms).**

| Etapa | Quando | Detalhe |
|-------|--------|---------|
| Embedding dos documentos | Build time | Títulos do corpus → `Qwen3-Embedding-0.6B` (1024-d, L2-norm) → `corpus_embeddings.npy` |
| Índice HNSW | Build time | `zvec` constrói grafo HNSW (métrica cosseno) e persiste em `artifacts/corpus/` |
| Embedding da query | Query time | `product_id` no corpus → vetor recuperado do `.npy` (sem inferência). Texto livre → uma chamada ao modelo com prefixo de instrução |
| ANN search | Query time | Busca HNSW em O(log N) → distâncias cosseno dos top-N candidatos |
| Similaridade | Query time | `sim = 1 − dist_cosseno` (vetores L2-norm: produto interno = cosseno) |
| Re-rank | Query time | `score = α · sim + (1−α) · S(p)`, α = 0.5 padrão |

## Fluxo de dados

```
Usuário clica num produto
  → GET /recommend?product_id=X
  → Backend: produto_id → embedding pré-computado do .npy
  → HNSW ANN search no corpus (< 10 ms)
  → Re-rank por S(p) = proporção de ratings ≥ 4
  → Enriquece com avg_rating e review_count do val+test
  → JSON → frontend renderiza score bars e fórmula explicativa
```

## Dados

| Conjunto | Origem | Papel no webapp |
|----------|--------|-----------------|
| **Storefront** | `storefront_catalog.csv` (artefato de 04_recommender) | Catálogo de navegação e busca |
| **Corpus** | `corpus_catalog.csv` + embeddings (artefato de 04_recommender) | Pool ANN para recomendações |
| **Reviews** | `B2W-Reviews01_val.csv` + `B2W-Reviews01_test.csv` | avg_rating, review_count e avaliações exibidas na página de produto (incluem rating=3) |

Para mais detalhes sobre decisões de design, ver [`handoff.md`](handoff.md).
