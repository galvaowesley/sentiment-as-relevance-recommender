# vectorizer — Embeddings de Títulos de Produtos

Vetoriza os títulos de produto (`product_name`) usando o **Serafim PT*** (`serafim-335m-portuguese-pt-sentence-encoder`), sentence encoder aberto para português com 335M de parâmetros.

Cada título é codificado em um vetor denso de **1024 dimensões** via mean pooling, adequado para busca por similaridade de cosseno no banco vetorial. Os vetores gerados são persistidos para indexação em `../vector_store/`.
