# bertimbau_ft — BERTimbau Fine-tuning + MLP

BERTimbau Base com **fine-tuning completo**: uma camada linear é adicionada ao [CLS] e toda a rede é ajustada ponta-a-ponta na tarefa de classificação binária de sentimento.

Configuração esperada como modelo campeão segundo a literatura (AUC-ROC > 0.95 em benchmarks de PT-BR). Os pesos finais são salvos em `../checkpoints/`.

**Validação:** k=3 folds.
