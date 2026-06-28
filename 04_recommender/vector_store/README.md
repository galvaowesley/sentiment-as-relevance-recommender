# vector_store — Banco Vetorial e Busca ANN

Indexa os embeddings de títulos gerados em `../vectorizer/` e realiza busca por
**Approximate Nearest Neighbors (ANN)** com similaridade de cosseno usando o
**zvec** (índice HNSW, métrica COSINE).

Dado o produto de consulta `q`, recupera os **Top-K** títulos mais similares do
índice (`K = retrieve_k` em `config.py`). O zvec reporta *distância* de cosseno;
o wrapper converte para similaridade com `sim = 1 − distância`.

A coleção é persistida em disco (`artifacts/corpus/`) e reaberta pelo serviço.
Implementação em `store.py` (`ZvecVectorStore`).
