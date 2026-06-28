# vectorizer — Embeddings de Títulos de Produtos

Vetoriza os títulos de produto (`product_name`) com o **Qwen3-Embedding-0.6B**
(`Qwen/Qwen3-Embedding-0.6B`), embedding multilíngue e instruction-aware.

Cada título vira um vetor denso de **1024 dimensões**, L2-normalizado (produto
interno = cosseno), adequado à busca por similaridade no banco vetorial.

- **Documentos** (corpus) são codificados sem instrução.
- **Consultas** (produto da página) recebem o prefixo de instrução recomendado
  para o Qwen3, seguindo o setup de retrieval assimétrico.

O device é escolhido automaticamente (CUDA → MPS → CPU). Implementação em
`embedder.py` (`Qwen3Embedder`).
