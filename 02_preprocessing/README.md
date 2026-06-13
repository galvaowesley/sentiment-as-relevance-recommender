# 02_preprocessing — Pré-processamento Textual

Transforma o texto bruto do corpus em representações prontas para os modelos. Organizado em duas camadas conforme o tipo de representação utilizada.

## Conteúdo

| Pasta | Descrição |
|---|---|
| `common/` | Normalização compartilhada entre todos os modelos |
| `tfidf/` | Etapas adicionais específicas para a representação TF-IDF |

## Por que duas camadas?

O BERTimbau usa seu próprio tokenizador WordPiece e requer apenas a normalização básica de `common/`. O pipeline TF-IDF precisa de etapas extras (stopwords, lematização, n-gramas) que estão isoladas em `tfidf/` para não poluir o pré-processamento do BERT.
