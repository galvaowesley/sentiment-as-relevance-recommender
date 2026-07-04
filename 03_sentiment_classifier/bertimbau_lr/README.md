# bertimbau_lr — BERTimbau [CLS] + Regressão Logística (Campeão)

Usa o BERTimbau Base (`neuralmind/bert-base-portuguese-cased`, 110M parâmetros) como extrator de features com **pesos congelados**. O vetor do token `[CLS]` da última camada (768 dimensões) é extraído para cada review e usado como entrada de um classificador de Regressão Logística externo.

É o **modelo campeão** em uso: sua inferência (`eval_bertimbau_lr.py`) gera o CSV `B2W-Reviews01_inferred_bertimbau.csv`, promovido para `data/results_from_best_model/` e consumido pelo recomendador (Pipeline 2) para calcular o score de positividade `S(p)`.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `train_bertimbau_lr.py` | Extração de embeddings `[CLS]`, validação cruzada, avaliação e salvamento |
| `eval_bertimbau_lr.py` | Inferência do modelo treinado → CSV enriquecido (`--dataset ""\|no_neutral`) |

## Pipeline (treino)

1. Carrega os splits sem neutros (`data/processed/B2W-Reviews01_no_neutral_*.csv`)
2. Normalização BERT via `02_preprocessing/bert/tokenize.py` (sem lowercase, sem remoção de acentos)
3. Extração dos embeddings `[CLS]` com pesos congelados em batches de 32 — **resultado cacheado em disco** como `.npy`
4. Grid Search com k=3 StratifiedKFold sobre os embeddings do treino
5. Avaliação independente no val e no test
6. Salva artefatos em `03_sentiment_classifier/checkpoints/`

## Cache de embeddings

A extração é cara (~30 min em CPU). Na primeira execução os embeddings são salvos em `checkpoints/bert_cls_{train,val,test}_X.npy` (+ `_y.npy`) e reaproveitados em execuções seguintes. Os `.npy` **não são versionados** (ver `.gitignore`).

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
| `checkpoints/bert_cls_*_X.npy` | Embeddings `[CLS]` cacheados por split (não versionados) |
| `checkpoints/bertimbau_lr_model.pkl` | Melhor modelo de Regressão Logística |
| `checkpoints/bertimbau_lr_results.json` | Métricas no val e no test (coletânea curada em `../train_evaluation/`) |
| `../inference/outputs/B2W-Reviews01_inferred_bertimbau.csv` | CSV enriquecido (colunas `inferencia_bertimbau`, `probabilidade_bertimbau`) |

## Execução

```bash
# via Makefile na raiz
make train-bertimbau
make infer-bertimbau
make champion           # treino + inferência + promove o CSV para data/results_from_best_model/

# ou manualmente
python 03_sentiment_classifier/bertimbau_lr/train_bertimbau_lr.py
python 03_sentiment_classifier/bertimbau_lr/eval_bertimbau_lr.py --dataset ""
```
