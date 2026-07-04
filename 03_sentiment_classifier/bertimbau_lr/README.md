# bertimbau_lr — BERTimbau [CLS] + Regressão Logística

Usa o BERTimbau Base (110M parâmetros) como extrator de features com **pesos congelados**. O vetor do token `[CLS]` da última camada (768 dimensões) é extraído para cada review e usado como entrada de um classificador de Regressão Logística externo.

Serve como ponto intermediário entre o baseline esparso (TF-IDF) e o fine-tuning completo, isolando a contribuição da representação contextual pré-treinada sem ajuste supervisionado.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `train_bertimbau_lr.py` | Script completo de extração de embeddings, validação cruzada, avaliação e salvamento |
| `bertimbau.py` | Snippet de referência para carregamento do modelo |

## Pipeline

1. Carrega os splits sem neutros (`data/processed/B2W-Reviews01_no_neutral_*.csv`)
2. Normalização BERT via `02_preprocessing/bert/tokenize.py` (sem lowercase, sem remoção de acentos)
3. Extração dos embeddings `[CLS]` com pesos congelados em batches de 32 — **resultado cacheado em disco** como `.npy`
4. Grid Search com k=3 StratifiedKFold sobre os embeddings do treino
5. Avaliação independente no val e no test
6. Salva artefatos em `03_sentiment_classifier/checkpoints/`

## Cache de embeddings

A extração é cara (~30 min em CPU para 70k amostras). Na primeira execução os embeddings são salvos em:

```
checkpoints/bert_cls_train_X.npy  /  bert_cls_train_y.npy
checkpoints/bert_cls_val_X.npy    /  bert_cls_val_y.npy
checkpoints/bert_cls_test_X.npy   /  bert_cls_test_y.npy
```

Execuções subsequentes carregam diretamente do cache, pulando o BERT.

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
| `checkpoints/bert_cls_*_X.npy` | Embeddings [CLS] cacheados por split |
| `checkpoints/bertimbau_lr_model.pkl` | Melhor modelo de Regressão Logística |
| `checkpoints/bertimbau_lr_results.json` | Métricas de avaliação no val e no test |

## Execução

```bash
python 03_sentiment_classifier/bertimbau_lr/train_bertimbau_lr.py
```
