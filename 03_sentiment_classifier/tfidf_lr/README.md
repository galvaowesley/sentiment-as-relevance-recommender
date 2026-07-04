# tfidf_lr — TF-IDF + Regressão Logística (Baseline)

Modelo de linha de base com representação esparsa. Usa a matriz TF-IDF produzida em `02_preprocessing/tfidf/` como entrada para um classificador de Regressão Logística com pesos por classe para lidar com o desbalanceamento do corpus.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `train_tfidf.py` | Treino, validação cruzada, avaliação e salvamento de artefatos (`--ngrams unigrams\|bigrams\|both`) |
| `eval_tfidf.py` | Inferência do modelo treinado → CSV enriquecido (`--ngrams …`, `--dataset ""\|no_neutral`) |
| `all_ngrams.sh` | Treina as três variantes de n-grama (rodar de dentro desta pasta) |
| `all_ngrams_eval.sh` | Roda a inferência para as três variantes (rodar de dentro desta pasta) |

## Pipeline (treino)

1. Carrega os splits sem neutros (`data/processed/B2W-Reviews01_no_neutral_*.csv`)
2. Normalização textual comum via `02_preprocessing/common/preprocess.py`
3. Lematização + vetorização TF-IDF via `02_preprocessing/tfidf/vectorize.py`
4. Grid Search com k=5 StratifiedKFold apenas sobre o conjunto de treino
5. Avaliação independente no val e no test (F1-macro, AUC-ROC, Precisão, Revocação, Matriz de Confusão)
6. Salva artefatos em `03_sentiment_classifier/checkpoints/`

## Hiperparâmetros buscados

| Parâmetro | Valores |
|---|---|
| `C` | 0.01, 0.1, 1.0, 10.0 |
| `solver` | lbfgs, saga |
| `max_iter` | 1000 |
| `class_weight` | `{0: 2.5, 1: 1.0}` (fixo — favorece a classe negativa) |

## Artefatos gerados (`checkpoints/`, por variante de n-grama)

| Arquivo | Descrição |
|---|---|
| `tfidf_{ngrams}_vectorizer.pkl` | Vetorizador TF-IDF ajustado |
| `tfidf_{ngrams}_lr_model.pkl` | Melhor modelo de Regressão Logística |
| `tfidf_{ngrams}_lr_results.json` | Métricas no val e no test (coletânea curada em `../train_evaluation/`) |
| `../inference/outputs/B2W-Reviews01_inferred_tfidf_{ngrams}.csv` | CSV enriquecido (gerado por `eval_tfidf.py`) |

## Execução

```bash
# via Makefile na raiz
make train-tfidf NGRAMS=both
make infer-tfidf NGRAMS=both

# ou manualmente
python 03_sentiment_classifier/tfidf_lr/train_tfidf.py --ngrams both
python 03_sentiment_classifier/tfidf_lr/eval_tfidf.py --ngrams both --dataset ""
```
