# 01_eda — Análise Exploratória de Dados

Exploração do corpus B2W-Reviews01 antes de qualquer modelagem. O objetivo desta fase é entender a distribuição de classes, identificar ruídos e particularidades do domínio que informam as decisões de pré-processamento e modelagem.

## Conteúdo

| Pasta | Descrição |
|---|---|
| `notebooks/` | Notebooks Jupyter da exploração interativa (**planejado — ainda sem arquivos**) |
| `figures/` | Visualizações exportadas (**planejado — ainda sem arquivos**) |

> As figuras da EDA usadas no relatório estão atualmente em `docs/reports/pln_projeto_relatorio/figuras/`. Esta pasta é excluída da imagem Docker (`.dockerignore`).

## O que é coberto

- Distribuição das pontuações (1–5 estrelas) e mapeamento de polaridade binária
- Frequência dos termos mais recorrentes após remoção de stopwords
- Top categorias por volume de produtos e por média de avaliação
- Análise de dados faltantes nos campos `review_text` e `review_title`
- Comprimento médio dos textos por polaridade
- Nuvem de palavras por classe (positivo / negativo)
