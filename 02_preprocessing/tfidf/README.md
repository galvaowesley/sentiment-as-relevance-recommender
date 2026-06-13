# tfidf — Pré-processamento Específico para TF-IDF

Etapas adicionais aplicadas **somente** ao pipeline de representação esparsa, após a normalização de `common/`:

1. Tokenização por espaço e pontuação
2. Remoção de stopwords em português (lista NLTK)
3. Lematização com spaCy (`pt_core_news_sm`)
4. Construção do vocabulário e matriz TF-IDF com n-gramas de 1 a 2

Os bigramas capturam associações locais relevantes para o domínio ("não recebi", "boa qualidade", "prazo entrega") que se perdem quando os tokens são analisados isoladamente.
