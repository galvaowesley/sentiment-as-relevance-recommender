# checkpoints — Modelos e Predições Salvos

Vetorizadores, modelos treinados e predições produzidos pelo Pipeline 1.

## Conteúdo (versionado)

| Arquivo | Origem |
|---|---|
| `tfidf_{unigrams,bigrams,both}_vectorizer.pkl` | `tfidf_lr/train_tfidf.py` |
| `tfidf_{unigrams,bigrams,both}_lr_model.pkl` | `tfidf_lr/train_tfidf.py` |
| `bertimbau_lr_model.pkl` | `bertimbau_lr/train_bertimbau_lr.py` |
| `llm_*_predictions.csv` | `llm/infer_llm.py` (uma linha por review, resumível) |

> Os `.pkl` e os `llm_*_predictions.csv` **são versionados** (são pequenos). Apenas os embeddings `[CLS]` cacheados (`bert_cls_*_X.npy` / `_y.npy`) ficam de fora — o `.gitignore` ignora `*.npy`. Os JSONs de métricas escritos aqui pelos scripts têm sua coletânea curada em `../train_evaluation/`.
