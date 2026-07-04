# tests — Suíte pytest do Recomendador

Testes automatizados do Pipeline 2. Rápidos por padrão: usam um `FakeEmbedder` determinístico (definido em `conftest.py`) e constroem artefatos zvec reais em `tmp_path`, exercitando recuperação, re-ranqueamento, persistência e a API HTTP **sem baixar modelos**.

| Arquivo | Cobre |
|---|---|
| `test_data.py` | Carregamento/normalização do catálogo |
| `test_recommender.py` | Fluxo end-to-end de recomendação |
| `test_reranker.py` | Fórmula `Score = α·sim + (1−α)·S(p)` |
| `test_sentiment.py` | Cálculo do score `S(p)` |
| `test_vector_store.py` | Persistência/consulta no zvec |
| `test_service.py` | Endpoints FastAPI (`TestClient`) |
| `test_embedder.py` | Embedder real Qwen3 (marcado `integration`) |

## Execução

```bash
make test               # rápidos (exclui integration)
make test-integration   # baixa e roda o Qwen3-Embedding-0.6B real

# ou diretamente
cd 04_recommender && pytest
```
