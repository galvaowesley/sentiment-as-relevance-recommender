# bertimbau_lr — BERTimbau [CLS] + Regressão Logística

Usa o BERTimbau Base (110M parâmetros) como extrator de features com **pesos congelados**. O vetor do token [CLS] da última camada é extraído como representação de entrada para um classificador de Regressão Logística externo.

Serve como ponto intermediário entre o baseline esparso (TF-IDF) e o fine-tuning completo, isolando a contribuição da representação contextual sem ajuste supervisionado.

**Validação:** k=3 folds.
