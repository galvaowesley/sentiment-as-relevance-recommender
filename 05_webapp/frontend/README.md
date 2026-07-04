# frontend — Interface do Site Simulado (SentiShop)

SPA que simula um e-commerce e consome a API do `backend/`. Permite navegar por categorias, abrir um produto e visualizar as recomendações re-ranqueadas — cada item com barras de **similaridade** e **sentimento** que explicam a pontuação `Score = α·sim + (1−α)·S(p)`.

## Stack

React 18 + **Vite 6** + `react-router-dom` (JavaScript/JSX, sem TypeScript). Gerenciador de pacotes: **npm** (`package-lock.json`).

## Estrutura

- `index.html` → `src/main.jsx` → `src/App.jsx` (rotas `/` = Browse, `/product/:productId` = detalhe)
- `src/api.js` — cliente da API; base em `import.meta.env.VITE_API_URL`
- `src/components/` — `Header`, `CategorySidebar`, `CategoryNavBar`, `ProductCard`, `Pagination`, `RecommendationsPanel`, `SentimentBadge`, `CategoryIcon`

## Configuração da API

| Arquivo | `VITE_API_URL` | Uso |
|---|---|---|
| `.env` | `http://localhost:8000` | Dev — aponta para o FastAPI local |
| `.env.production` | *(vazio)* | Prod — mesma origem (o backend serve o SPA) |

## Execução

```bash
# via Makefile na raiz
make install-web        # npm ci
make frontend           # dev server em :5173
make build-web          # compila para dist/ (servido pelo backend em produção)

# ou diretamente
cd 05_webapp/frontend
npm run dev             # :5173
npm run build           # -> dist/
npm run preview         # pré-visualiza o build
```
