# llm — Classificação One-shot via LLM Local (LM Studio)

Abordagem sem treinamento: usa um LLM hospedado localmente via LM Studio (SDK Python `lmstudio`) para classificar o sentimento de cada avaliação diretamente por prompting, com exatamente um exemplo (one-shot) embutido no prompt de sistema. Ao contrário de `tfidf_lr/` e `bertimbau_lr/`, não há ajuste de nenhum classificador — o pipeline é de inferência + avaliação sobre val e test (nunca usados para treino).

## Arquivos

| Arquivo | Descrição |
|---|---|
| `prompt_one_shot.yaml` | Config autocontida do prompt: instruções de sistema, exemplo one-shot, template do turno do usuário e schema JSON da resposta |
| `infer_llm.py` | Roda inferência linha a linha sobre val+test, salva predições em CSV |
| `evaluate_llm.py` | Recalcula métricas por split a partir do CSV de predições e salva `*_results.json` |

## Pipeline

1. Carrega val + test sem neutros (`data/processed/B2W-Reviews01_no_neutral_{val,test}.csv`) — o LLM não é treinado, apenas avaliado nesses dois splits
2. Combina título+texto com limpeza leve (`fillna` + `strip`, sem a normalização agressiva usada por TF-IDF/BERTimbau)
3. Uma requisição independente por avaliação — contexto limpo a cada chamada (mensagens novas, nada reaproveitado entre linhas), saída estruturada via JSON schema
4. Salva CSV de predições em `checkpoints/`
5. `evaluate_llm.py` recalcula F1-macro/AUC-ROC/classification_report por split e salva `_results.json`

## Pré-requisitos

- LM Studio rodando localmente com um modelo já baixado
- Identificador do modelo: descubra com `lms ls` (CLI do LM Studio) ou `lmstudio.list_downloaded_models()`

## Retomada (resume)

`infer_llm.py` é seguro de interromper: cada linha é gravada no CSV assim que respondida (`row_id` no formato `val:0`, `test:0`, ...). Ao rodar de novo com o mesmo `--output`, linhas já resolvidas com sucesso são puladas; linhas que falharam anteriormente são descartadas do arquivo e tentadas novamente. Um smoke test com `--limit` pequeno seguido de um run completo sem `--limit` também funciona — apenas as linhas já concluídas são puladas.

## Artefatos gerados

| Arquivo | Descrição |
|---|---|
| `checkpoints/llm_<modelo>_predictions.csv` | Todas as colunas originais + `split`, `true_polarity`, `llm_raw_response`, `llm_predicted_polarity`, `llm_error`, tokens de entrada/saída e tempo de resposta por linha |
| `checkpoints/llm_<modelo>_results.json` | Métricas de avaliação no val e no test (mesmo formato de `tfidf_bigrams_lr_results.json`) |

## Execução

```bash
python 03_sentiment_classifier/llm/infer_llm.py --model "qwen/qwen3-4b-2507" --limit 10   # smoke test
python 03_sentiment_classifier/llm/infer_llm.py --model "qwen/qwen3-4b-2507"               # run completo (retomável)
python 03_sentiment_classifier/llm/evaluate_llm.py --predictions 03_sentiment_classifier/checkpoints/llm_qwen_qwen3-4b-2507_predictions.csv
```

## Limitações

- **Tempo de execução**: val+test somam ~46 mil avaliações; sem batching (por design, para garantir contexto limpo por item), um run completo local pode levar de horas a mais de um dia dependendo do modelo/hardware. Rode `--limit` primeiro para estimar o tempo total.
- **AUC-ROC**: como a classificação one-shot retorna apenas um rótulo (sem probabilidade calibrada), o AUC-ROC é calculado a partir da predição binária dura — equivale a uma acurácia balanceada, não diretamente comparável ao AUC-ROC de `tfidf_lr`/`bertimbau_lr` (que usam `predict_proba`).
- **`response_format` + mensagens system+user**: combinação validada empiricamente no primeiro run local — se o schema estruturado não for respeitado pelo modelo escolhido, a linha é registrada com `llm_error` preenchido e não interrompe o lote.
