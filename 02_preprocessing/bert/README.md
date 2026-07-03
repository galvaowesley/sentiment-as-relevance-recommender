# bert — Pré-processamento para BERTimbau

Módulo de tokenização e normalização para os modelos `bertimbau_lr` e `bertimbau_ft`.

## Por que não reutilizar `common/preprocess.py` diretamente?

| Etapa | TF-IDF | BERTimbau |
|---|---|---|
| Caixa baixa | sim | **não** — modelo é case-sensitive |
| Remoção de acentos | não (regex preserva) | **não** — acentos têm peso nos embeddings |
| Remoção de HTML/URL | sim | sim |
| Lematização | sim (spaCy) | **não** — WordPiece opera sobre tokens brutos |
| Stopwords | sim (NLTK) | **não** — BERT aprende a ignorar |

## Conteúdo

| Símbolo | Descrição |
|---|---|
| `normalize_text_bert(text)` | Normalização leve: remove HTML, URLs e colapsa espaços |
| `build_text_input_bert(df)` | Concatena `review_title` + `review_text` e normaliza |
| `load_tokenizer()` | Carrega `AutoTokenizer` do `neuralmind/bert-base-portuguese-cased` |
| `ReviewDataset` | `torch.utils.data.Dataset` com `input_ids`, `attention_mask` e `labels` |

## Parâmetros

- **Modelo**: `neuralmind/bert-base-portuguese-cased`
- **`MAX_LENGTH`**: 256 tokens (cobre >99% das reviews B2W; limite duro do BERT é 512)
- **Padding**: `max_length` (para batches homogêneos)
- **Truncamento**: habilitado
