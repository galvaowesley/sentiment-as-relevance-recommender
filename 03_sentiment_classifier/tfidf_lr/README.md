# tfidf_lr — TF-IDF + Regressão Logística (Baseline)

Modelo de linha de base com representação esparsa. Usa a matriz TF-IDF produzida em `02_preprocessing/tfidf/` como entrada para um classificador de Regressão Logística com pesos por classe para lidar com o desbalanceamento do corpus.

**Validação:** k=5 folds com otimização de hiperparâmetros via Grid Search (C, solver, max_iter).
