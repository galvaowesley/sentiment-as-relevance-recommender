# tfidf_lr — TF-IDF + Regressão Logística (Baseline)

Modelo de linha de base com representação esparsa. Usa a matriz TF-IDF produzida em `02_preprocessing/tfidf/` como entrada para um classificador de Regressão Logística com pesos por classe para lidar com o desbalanceamento do corpus.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `train_tfidf.py` | Script completo de treino, validação cruzada, avaliação e salvamento de artefatos |

## Pipeline

1. Carrega os splits sem neutros (`data/processed/B2W-Reviews01_no_neutral_*.csv`)
2. Normalização textual comum via `02_preprocessing/common/preprocess.py`
3. Lematização + vetorização TF-IDF via `02_preprocessing/tfidf/vectorize.py`
4. Grid Search com k=5 StratifiedKFold apenas sobre o conjunto de treino
5. Avaliação independente no val e no test (F1-macro, AUC-ROC, Precisão, Revocação, Matriz de Confusão)
6. Salva artefatos em `03_sentiment_classifier/checkpoints/`

## Validação

O Grid Search com k=5 folds roda somente sobre o conjunto de treino. Val e test são mantidos separados e avaliados de forma independente como dois conjuntos de teste distintos, permitindo comparar a generalização do modelo entre os dois splits.

## Hiperparâmetros buscados

| Parâmetro | Valores |
|---|---|
| `C` | 0.01, 0.1, 1.0, 10.0 |
| `solver` | lbfgs, saga |
| `max_iter` | 1000 |
| `class_weight` | balanced (fixo) |

## Artefatos gerados

| Arquivo | Descrição |
|---|---|
| `checkpoints/tfidf_vectorizer.pkl` | Vetorizador TF-IDF ajustado ao train+val |
| `checkpoints/tfidf_lr_model.pkl` | Melhor modelo de Regressão Logística |
| `checkpoints/tfidf_lr_results.json` | Métricas de avaliação no teste |

## Execução

```bash
python 03_sentiment_classifier/tfidf_lr/train_tfidf.py
```
