# backend — API REST

Serviço que expõe os endpoints de recomendação. Ao iniciar, carrega o modelo campeão de sentimento (`03_sentiment_classifier/checkpoints/`) e o índice vetorial (`04_recommender/vector_store/`).

## Principal endpoint

```
POST /recommend
Body: { "product_name": "...", "k": 10, "alpha": 0.6 }
Response: [{ "product_name": "...", "similarity": 0.92, "sentiment_score": 0.87, "final_score": 0.90 }, ...]
```

## Stack sugerida

FastAPI + Uvicorn (leve, assíncrono, geração automática de docs em `/docs`).
