# SentiShop — Handoff: Metodologia do Site de Demonstração

## Visão geral

O SentiShop é uma demo web do pipeline de **recomendação semântica com re-rank por sentimento**. O objetivo é ser explicável: o usuário vê *como* cada recomendação foi gerada — não apenas o resultado, mas os dois estágios do pipeline e a fórmula do score final.

---

## Arquitetura do pipeline de recomendação

```
Consulta (produto ou texto livre)
        │
        ▼
 ① Recuperação por similaridade
    - Busca vetorial no corpus (embeddings)
    - Retorna os top-N mais similares (score de similaridade textual)
        │
        ▼
 ② Re-rank por sentimento
    - Cada candidato recebe S(p) = proporção de avaliações positivas (overall_rating ≥ 4)
    - score_final = α · sim + (1 − α) · S(p)    [α = 0.5 por padrão]
    - Reordena os candidatos pelo score_final
        │
        ▼
 Resultado: produtos re-rankados apresentados com os três scores visíveis
```

---

## Como é calculada a similaridade?

### Resposta curta

> **Os embeddings são pré-computados (offline). A busca de vizinhos acontece em tempo real, mas é rápida (ms) graças ao índice HNSW.**

### Detalhes

| Etapa | Quando | O que acontece |
|-------|--------|----------------|
| **Embedding dos documentos** | Build time (offline) | Títulos dos produtos do corpus são codificados com `Qwen3-Embedding-0.6B` sem prefixo de instrução; vetores (1024-d, float32, L2-normalizados) salvos em `corpus_embeddings.npy` |
| **Construção do índice HNSW** | Build time (offline) | `zvec` constrói um grafo HNSW com métrica cosseno sobre os vetores salvos; índice persiste em `artifacts/corpus/` |
| **Embedding da query** | Query time — só se necessário | Se a query for um `product_id` já no corpus → embedding recuperado diretamente do `.npy` (sem inferência). Se for texto livre → o modelo é chamado uma vez, com prefixo de instrução (`encode_queries`) |
| **ANN search** | Query time — sempre | `zvec` executa busca ANN no grafo HNSW; retorna top-N candidatos com distância cosseno em O(log N) — tipicamente < 10 ms |
| **Similaridade** | Query time | `sim = 1 − distância_cosseno` (vetores L2-normalizados: produto interno = cosseno) |
| **Re-rank** | Query time | `score = α · sim + (1−α) · S(p)` com α = 0.5 padrão |

### Por que cosseno?

O modelo usa retrieval **assimétrico**: corpus codificado sem instrução, query codificada com prefixo. Isso segue a recomendação do Qwen3-Embedding para tasks de recuperação. Como os vetores são L2-normalizados, o produto interno é numericamente equivalente ao cosseno — `zvec` usa `COSINE` como `MetricType`.

### Implicação para escala

Com o corpus atual (~2k produtos), a busca é instantânea mesmo sem índice. Ao escalar para todo o val+test (~50k produtos), o HNSW continua sub-linear e sem necessidade de reembedding.

---

## Splits de dados e seus papéis

| Split | Uso | Inclui neutros (rating=3)? |
|-------|-----|---------------------------|
| `B2W-Reviews01_train.csv` | treino do vetorizador / corpus vetorizado | **Não** (apenas 1, 2, 4, 5) |
| `B2W-Reviews01_no_neutral_train.csv` | idem, explicitamente sem neutros | Não |
| `B2W-Reviews01_val.csv` | corpus do storefront (webapp) + reviews | **Sim** (vida real) |
| `B2W-Reviews01_test.csv` | idem | **Sim** |
| `B2W-Reviews01_no_neutral_val/test.csv` | versões sem neutros (para avaliação de classificador) | Não |

**Decisão de design**: o corpus do motor de recomendação será construído a partir de `val` + `test` (com neutros), pois representa melhor o catálogo real. O treino sem neutros mantém o vetorizador treinado apenas em opiniões polares.

---

## Estrutura dos artefatos (`04_recommender/artifacts/`)

| Arquivo | Conteúdo |
|---------|----------|
| `storefront_catalog.csv` | Catálogo de produtos exibidos no webapp (product_id, product_name, brand, lv1, lv2, num_reviews, sentiment_score) |
| `corpus_catalog.csv` | Catálogo dos produtos buscáveis/recomendáveis (subconjunto vetorizado) |
| `storefront_embeddings.npy` | Embeddings do storefront |
| `corpus_embeddings.npy` | Embeddings do corpus |
| `meta.json` | Metadados do índice (modelo, alpha, dimensões) |

---

## Backend (FastAPI — `05_webapp/backend/app.py`)

### Endpoints principais

| Endpoint | Descrição |
|----------|-----------|
| `GET /categories` | Lista categorias lv1 do storefront |
| `GET /categories/lv2?cat1=X` | Subcategorias lv2 para um lv1 |
| `GET /products` | Catálogo paginado e filtrável (suporta `category`, `cat2`, `q`) |
| `GET /products/count` | Total de produtos com os mesmos filtros — usado pela paginação |
| `GET /products/{id}` | Produto com `avg_rating`, `review_count`, `description` enriquecidos |
| `GET /products/{id}/reviews` | Todas as avaliações do produto (val+test, inclusive rating=3) |
| `GET /recommend` | Top-N recomendações re-rankadas por sentimento |

### Enriquecimento de dados no startup

O backend carrega `val.csv` + `test.csv` na inicialização e constrói:
- `_reviews_by_product`: dict `{product_id → list[dict]}` com todas as avaliações
- `_product_stats`: dict `{product_id → {avg_rating, review_count}}`

Esses dados são injetados em todos os endpoints de produto via `_enrich_product()`.

**"Descrição"**: o dataset B2W não tem campo de descrição de produto. A seção "Sobre o produto" exibe apenas metadados reais (`Marca: X. Categoria: lv1 › lv2.`). Texto de avaliação (review_text) **nunca** é usado como descrição — seria uma opinião, não uma descrição factual. Geração real via LLM está prevista como próxima iteração.

---

## Frontend (React + Vite — `05_webapp/frontend/`)

### Componentes chave

| Componente | Responsabilidade |
|------------|-----------------|
| `CategorySidebar` | Sidebar esquerda colapsável: lv1 como lista, lv2 como accordion lazy-loaded; 220px expandida / 40px colapsada |
| `Browse` | Layout flex (sidebar + grid); 100 produtos/página; chama `/products/count` para total real |
| `Pagination` | Botões de número de página com window ±2 + reticências; mostra "X produtos · Y páginas" |
| `ProductCard` | Card com ícone, lv2 label, nome, brand, estrelas + avg_rating + review_count |
| `ProductDetail` | Breadcrumb clicável (lv1/lv2), info card, seção "Sobre o produto" (metadados), reviews com filtros, painel de recomendações |
| `RecommendationsPanel` | Pipeline steps badge, rec cards com score bars (teal=similaridade, red=sentimento), fórmula |
| `ReviewItem` (inline) | Avaliação individual: rating, texto, recommend badge, `S(p)` badge (pendente ou calculado) |

### Lógica de score visual

Os três scores de cada recomendação são exibidos de forma distinta:
- **Similaridade textual** → barra teal (`#0D9488`) com valor decimal
- **Sentimento S(p)** → barra vermelha (`#DC2626`) com valor em percentual  
- **Score final** → número em destaque no canto superior direito de cada card de recomendação

### Filtros de avaliações

Na página de produto, o usuário pode filtrar as avaliações por:
- Recomenda / Não recomenda / Todos
- Estrelas (1★ a 5★)

### Sentimento por avaliação individual

Cada avaliação exibe um badge `S(p) pendente` (dashed border, cinza) quando o score individual ainda não foi calculado pelo pipeline de classificação. Quando disponível (campo `sentiment_score` não-null vindo da API), muda para badge colorido (verde/vermelho). Isso prepara a interface para a próxima iteração sem precisar refatorar.

---

## Próximas iterações planejadas

1. **Pipeline de classificação**: calcular `sentiment_score` por avaliação individual (campo já reservado na API e na UI) usando o classificador treinado
2. **Corpus maior**: rebuildar o índice vetorial a partir de val+test (atualmente ~2k itens; escalar para todos os produtos com ≥ N avaliações)
3. **Descrições via LLM**: gerar descrição real do produto condicionada ao `product_name` + `brand` + avaliações (substituir a seção "Sobre o produto" que hoje exibe apenas metadados)
4. **Score S(p) dinâmico**: quando a classificação estiver disponível, recalcular S(p) com os scores individuais em vez da proporção binária de ratings
5. **Alpha ajustável**: expor o parâmetro α na UI para o usuário experimentar o tradeoff similaridade × sentimento
