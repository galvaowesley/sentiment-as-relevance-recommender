# vector_store — Banco Vetorial e Busca ANN

Indexa os embeddings de títulos gerados em `../vectorizer/` e realiza a busca por **Approximate Nearest Neighbors (ANN)** com similaridade de cosseno.

**Candidatos de banco vetorial:** FAISS, ChromaDB ou Qdrant (a definir na etapa de implementação).

Dado um produto de consulta q, o sistema recupera os Top-K títulos mais similares do índice. O valor de K é definido empiricamente durante a validação do módulo de recomendação.
