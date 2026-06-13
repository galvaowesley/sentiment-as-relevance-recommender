# common — Normalização Compartilhada

Etapas de limpeza textual aplicadas a **todos** os modelos (TF-IDF e BERTimbau):

1. Conversão para minúsculas
2. Remoção de HTML tags, URLs e caracteres especiais não-alfanuméricos
3. Normalização de acentuação e caracteres Unicode (NFD/NFC)
4. Remoção de espaços duplicados e linhas vazias

A entrada é o campo `review_text` concatenado ao `review_title`. A saída é consumida por `02_preprocessing/tfidf/` (para TF-IDF) e diretamente pelos scripts de treinamento do BERTimbau.
