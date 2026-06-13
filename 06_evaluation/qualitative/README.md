# qualitative — Avaliação Qualitativa

Avaliação humana independente conduzida pelos três integrantes do grupo, cobrindo ambos os pipelines.

## Pipeline 1 — Classificação

Amostra aleatória estratificada de **90 instâncias** do conjunto de teste avaliadas manualmente quanto à polaridade real do texto. Os rótulos humanos são comparados às predições do modelo campeão.

## Pipeline 2 — Recomendação

Cada integrante avalia de forma independente **20 pares consulta-lista** julgando se as recomendações são:
- Semanticamente relevantes para o produto de consulta
- De alta qualidade percebida (score de sentimento elevado)
- Diversas o suficiente para agregar valor

## Concordância

Em ambos os casos, a concordância entre avaliadores é medida pelo **Kappa de Cohen**.
