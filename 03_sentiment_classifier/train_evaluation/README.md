# train_evaluation — Métricas dos Modelos

Coletânea curada dos JSONs de métricas (F1-macro, AUC-ROC, precisão/revocação por classe, melhores hiperparâmetros) de cada configuração de modelo. É uma pasta de artefatos, sem código.

| Arquivo | Modelo |
|---|---|
| `tfidf_{unigrams,bigrams,both}_lr_results.json` | TF-IDF + LR (por variante de n-grama) |
| `bertimbau_lr_results.json` | BERTimbau `[CLS]` + LR (campeão) |
| `llm_*_results.json` | Classificador generativo one-shot (LM Studio) |

> Os scripts de treino escrevem o JSON em `../checkpoints/`; esta pasta guarda a versão curada usada na comparação entre modelos e no relatório. Os JSONs do LLM são gerados por `llm/evaluate_llm.py` (`make eval-llm`).
