# classification — Avaliação do Pipeline 1

Métricas de avaliação dos modelos de classificação de sentimento no conjunto de teste, além de uma amostra qualitativa para inspeção manual das predições.

## Métricas

| Métrica | Descrição |
|---|---|
| Precisão | Proporção de predições positivas corretas |
| Revocação | Proporção de positivos verdadeiros recuperados |
| F1-macro | Média harmônica de P e R sem ponderação por frequência de classe |
| AUC-ROC | Área sob a curva ROC, independente do limiar de decisão |
| Matriz de confusão | Análise dos tipos de erro (FP e FN) |

Os valores por modelo são versionados em `03_sentiment_classifier/train_evaluation/` (um JSON por modelo). O modelo com melhor F1-macro é declarado **campeão** e promovido ao Pipeline 2.

## Amostra qualitativa

| Arquivo | Descrição |
|---|---|
| `sampled_50_B2W-Reviews01_inferred_bertimbau.csv` / `.json` | 50 avaliações amostradas das inferências do campeão (BERTimbau), para revisão manual |
