# 05_webapp — Demonstração Web

Aplicação web que demonstra o sistema de recomendação orientado por sentimento em uma interface que simula um site de e-commerce. Integra os dois pipelines do projeto através de uma API REST.

## Conteúdo

| Pasta | Responsabilidade |
|---|---|
| `backend/` | API REST que serve as recomendações |
| `frontend/` | Interface do site simulado de e-commerce |

## Fluxo

1. O usuário busca um produto no frontend
2. O frontend chama a API do backend com o produto de consulta
3. O backend vetoriza a consulta, busca candidatos no índice e re-ranqueia por sentimento
4. O frontend exibe a lista de recomendações com scores de sentimento

## Como executar

Consulte os READMEs de `backend/` e `frontend/` para instruções de instalação e execução.
