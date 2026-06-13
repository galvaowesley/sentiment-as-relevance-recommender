# 01_eda — Análise Exploratória de Dados

Exploração do corpus B2W-Reviews01 antes de qualquer modelagem. O objetivo desta fase é entender a distribuição de classes, identificar ruídos e particularidades do domínio que informam as decisões de pré-processamento e modelagem.

## Conteúdo

| Pasta | Descrição |
|---|---|
| `notebooks/` | Notebooks Jupyter com a exploração interativa |
| `figures/` | Visualizações exportadas, usadas no relatório e nas apresentações |

## O que é coberto

- Distribuição das pontuações (1–5 estrelas) e mapeamento de polaridade binária
- Frequência dos 20 termos mais recorrentes após remoção de stopwords
- Top 10 categorias por volume de produtos e por média de avaliação
- Análise de dados faltantes nos campos `review_text` e `review_title`
- Comprimento médio dos textos por polaridade
- Nuvem de palavras por classe (positivo / negativo)
