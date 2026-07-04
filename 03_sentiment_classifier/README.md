# 03_sentiment_classifier — Classificação de Sentimento (Pipeline 1)

Implementa, treina e compara diferentes configurações de modelo para classificação binária de sentimento (positivo / negativo) sobre o corpus B2W-Reviews01. O modelo com melhor F1-macro no teste é o **modelo campeão**, cujas inferências alimentam o Pipeline 2. Atualmente o campeão em uso é o **BERTimbau [CLS] + LR** (`bertimbau_lr/`).

## Conteúdo

| Pasta / arquivo | Papel | Representação |
|---|---|---|
| `tfidf_lr/` | Regressão Logística | TF-IDF (uni/bi-gramas) — baseline |
| `bertimbau_lr/` | Regressão Logística | BERTimbau `[CLS]` congelado (campeão) |
| `bertimbau_ft/` | MLP | BERTimbau fine-tuning completo (**planejado — sem código**) |
| `llm/` | LLM one-shot | Classificador generativo *training-free* via LM Studio |
| `inference/` | — | CSVs enriquecidos gerados pela inferência dos modelos |
| `train_evaluation/` | — | JSONs de métricas (F1-macro, AUC-ROC) de cada modelo |
| `checkpoints/` | — | Vetorizadores/modelos treinados (`.pkl`) e predições do LLM |
| `sample_data.py` | — | Amostragem qualitativa a partir dos CSVs inferidos |

## Como executar (via Makefile na raiz)

```bash
make train-tfidf NGRAMS=both        # baseline TF-IDF + LR
make champion                       # BERTimbau: treino + inferência + promove o CSV campeão
make classify-llm LLM_MODEL="qwen/qwen3-4b-2507" LIMIT=50   # LLM (LM Studio rodando)
make eval-llm     LLM_MODEL="qwen/qwen3-4b-2507"
```

## Divisão dos dados

Splits gerados por `02_preprocessing/split_data*.py` (padrão 60/20/20, estratificado). O **treino** usa os splits `no_neutral`; a **inferência** roda sobre os splits com neutros para enriquecer todas as avaliações (inclusive rating 3) consumidas pelo recomendador.

## Validação cruzada

- TF-IDF: k=5 folds com Grid Search
- BERTimbau: k=3 folds (custo computacional maior)
- LLM: sem treino/validação cruzada (classificação one-shot)
