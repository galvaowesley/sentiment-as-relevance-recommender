# inference — Saídas de Inferência

Armazena os CSVs enriquecidos produzidos pela inferência dos classificadores (não há código aqui; é uma pasta de artefatos).

## `outputs/`

| Arquivo | Origem |
|---|---|
| `B2W-Reviews01_inferred_bertimbau.csv` | `bertimbau_lr/eval_bertimbau_lr.py` — **campeão** (colunas `inferencia_bertimbau`, `probabilidade_bertimbau`) |
| `B2W-Reviews01_inferred_tfidf_{unigrams,bigrams,both}.csv` | `tfidf_lr/eval_tfidf.py` |
| `B2W-Reviews01_inferred_*_predictions_test_plus_val.csv` | Consolidação das predições do LLM (`llm/`) |

> O CSV do campeão é promovido para `data/results_from_best_model/` (via `make champion`) e consumido pelo recomendador. `sample_data.py` lê desta pasta para gerar amostras qualitativas.
