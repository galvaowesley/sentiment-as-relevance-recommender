# bertimbau_ft — BERTimbau Fine-tuning + MLP (planejado)

> **Status: planejado — ainda sem código.** Esta pasta contém apenas este README. O modelo campeão atualmente em uso é o `bertimbau_lr/` ([CLS] congelado + Regressão Logística).

Configuração prevista: BERTimbau Base com **fine-tuning completo** — uma cabeça linear/MLP sobre o `[CLS]` e toda a rede ajustada ponta-a-ponta na classificação binária de sentimento, com validação k=3. Esperado, segundo a literatura de PT-BR, como o modelo de maior desempenho (AUC-ROC > 0.95 em benchmarks). Quando implementado, os pesos finais devem ser salvos em `../checkpoints/`.
