# 03_sentiment_classifier — Classificação de Sentimento (Pipeline 1)

Implementa, treina e compara três configurações de modelo para classificação binária de sentimento (positivo / negativo) sobre o corpus B2W-Reviews01. O modelo com melhor F1-macro no conjunto de teste é selecionado como **modelo campeão** e consumido pelo Pipeline 2.

## Conteúdo

| Pasta | Modelo | Representação |
|---|---|---|
| `tfidf_lr/` | Regressão Logística | TF-IDF com uni e bigramas (baseline) |
| `bertimbau_lr/` | Regressão Logística | BERTimbau [CLS] com pesos congelados |
| `bertimbau_ft/` | MLP | BERTimbau com fine-tuning completo |
| `checkpoints/` | — | Pesos e artefatos dos modelos treinados |

## Divisão dos dados

- Treino: 70% (estratificado por classe)
- Validação: 15% (ajuste de hiperparâmetros)
- Teste: 15% (avaliação final, utilizado apenas uma vez)

## Validação cruzada

- TF-IDF: k=5 folds com Grid Search
- BERTimbau: k=3 folds (custo computacional maior)
