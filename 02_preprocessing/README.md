# 02_preprocessing — Pré-processamento Textual e Splits

Transforma o texto bruto do corpus em representações prontas para os modelos e gera os splits treino/val/teste. Organizado em camadas conforme o tipo de representação utilizada.

## Conteúdo

| Item | Descrição |
|---|---|
| `common/` | Normalização compartilhada por todos os modelos (`build_text_input`, `map_polarity`) |
| `tfidf/` | Etapas extras da representação TF-IDF: stopwords, lematização (spaCy `pt_core_news_sm`) |
| `bert/` | Tokenização WordPiece p/ BERTimbau (`neuralmind/bert-base-portuguese-cased`, normalização leve) |
| `split_data.py` | Gera `B2W-Reviews01_{train,val,test}.csv` (mantém avaliações neutras, rating 3) |
| `split_data_no_neutral.py` | Gera `B2W-Reviews01_no_neutral_{train,val,test}.csv` (descarta rating 3) |

> `common/`, `tfidf/` e `bert/` são **módulos de biblioteca** (sem CLI), importados pelos scripts de `03_sentiment_classifier/`. Apenas os `split_data*.py` são executáveis.

## Por que várias camadas?

O BERTimbau usa seu próprio tokenizador WordPiece e requer apenas normalização leve (`bert/`, sem lowercase nem remoção de acentos). O pipeline TF-IDF precisa de etapas extras (stopwords, lematização, n-gramas) isoladas em `tfidf/` para não interferir no pré-processamento do BERT. A normalização básica comum (limpeza de HTML/URL, NFC) vive em `common/`.

## Splits

```bash
# via Makefile (raiz)
make splits

# ou manualmente
python 02_preprocessing/split_data.py --input data/raw/B2W-Reviews01.csv --output-dir data/processed
python 02_preprocessing/split_data_no_neutral.py --input data/raw/B2W-Reviews01.csv --output-dir data/processed
```

Proporção padrão: 60% treino / 20% validação / 20% teste, estratificada (ajustável por flags `--train/--val/--test/--stratify/--seed`). O **treino** dos modelos usa os splits `no_neutral`; a **inferência** usa os splits com neutros para cobrir todas as avaliações.
