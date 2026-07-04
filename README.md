# Sentimento como Relevância: Recomendação de Produtos em E-commerce Brasileiro

Projeto da disciplina de Processamento de Linguagem Natural — PPGCC/UFSCar.

**Autores:** Cilene Renata Real · Jayme Sakae dos Reis Furuyama · Wesley Nogueira Galvão

---

## Sobre o projeto

Sistema integrado de duas etapas para recomendação de produtos em e-commerce em português brasileiro:

1. **Classificador de sentimento** — compara TF-IDF + Regressão Logística, BERTimbau (`[CLS]` congelado + LR) e um classificador generativo *training-free* (LLM one-shot via LM Studio) sobre o corpus B2W-Reviews01. O modelo campeão rotula cada avaliação como positiva/negativa.
2. **Motor de recomendação** — recupera produtos semanticamente similares pelo título (embeddings **Qwen3-Embedding-0.6B** + busca vetorial ANN com **zvec/HNSW**) e re-ranqueia os resultados por um score de positividade `S(p)` derivado do modelo campeão.

O elo entre as etapas é o CSV `data/results_from_best_model/B2W-Reviews01_inferred_bertimbau.csv`, que o recomendador consome para calcular `S(p)`.

---

## Estrutura do repositório

```
sentiment-as-relevance-recommender/
│
├── Makefile                   # Orquestra todo o fluxo (make help)
├── requirements.txt           # Dependências Python (unificadas)
├── Dockerfile / railway.toml  # Build/deploy do webapp (porta única)
│
├── 01_eda/                    # Análise exploratória do corpus B2W-Reviews01
│   ├── notebooks/             # Notebooks Jupyter de exploração (planejado)
│   └── figures/               # Visualizações geradas (planejado)
│
├── 02_preprocessing/          # Pré-processamento textual e splits
│   ├── common/                # Normalização compartilhada (build_text_input, map_polarity)
│   ├── tfidf/                 # Extras TF-IDF: stopwords, lematização (spaCy pt)
│   ├── bert/                  # Tokenização WordPiece p/ BERTimbau
│   ├── split_data.py          # Split treino/val/teste (mantém neutros)
│   └── split_data_no_neutral.py  # Split sem avaliações neutras (rating 3)
│
├── 03_sentiment_classifier/   # Pipeline 1: classificação de sentimento
│   ├── tfidf_lr/              # TF-IDF + Regressão Logística (baseline)
│   ├── bertimbau_lr/          # BERTimbau [CLS] congelado + LR (campeão)
│   ├── bertimbau_ft/          # BERTimbau fine-tuning + MLP (planejado, sem código)
│   ├── llm/                   # Classificador generativo one-shot (LM Studio)
│   ├── inference/             # CSVs enriquecidos (saídas de inferência)
│   ├── train_evaluation/      # JSONs de métricas dos modelos
│   ├── checkpoints/           # Vetorizadores/modelos treinados (.pkl) e predições
│   └── sample_data.py         # Amostragem qualitativa de inferências
│
├── 04_recommender/            # Pipeline 2: recomendação orientada por sentimento
│   ├── vectorizer/            # Embeddings de títulos com Qwen3-Embedding-0.6B
│   ├── vector_store/          # Indexação e busca ANN (zvec — HNSW, cosseno)
│   ├── reranking/             # Re-ranqueamento ponderado por sentimento S(p)
│   ├── service/               # API FastAPI standalone do recomendador
│   ├── build_index.py         # Constrói o corpus vetorial (artifacts/)
│   └── demo.py                # Demo end-to-end via linha de comando
│
├── 05_webapp/                 # Demo "SentiShop": e-commerce simulado
│   ├── backend/               # API REST FastAPI (embrulha o recomendador)
│   └── frontend/              # Interface web (React 18 + Vite)
│
├── 06_evaluation/             # Avaliação quantitativa
│   ├── classification/        # F1-macro, AUC-ROC, matriz de confusão
│   └── recommendation/        # NDCG@k, Precision@k, Recall@k
│
├── data/
│   ├── raw/                   # B2W-Reviews01.csv (não versionado — ver instruções)
│   ├── processed/             # Splits treino/val/teste (.csv) e scores de sentimento
│   ├── results_from_best_model/  # CSV inferido pelo campeão (consumido pelo recomendador)
│   └── samples/               # Amostras pequenas para testes rápidos
│
└── docs/
    ├── reports/               # Relatório parcial e projeto LaTeX do relatório final
    └── presentations/         # Apresentações dos seminários (planejado)
```

> Cada pasta contém um `README.md` descrevendo seu propósito e conteúdo.

---

## Execução com Makefile

O `Makefile` na raiz é o ponto de entrada único. Veja todos os alvos com:

```bash
make help
```

Fluxos principais:

```bash
# 1. Instalar dependências (Python + modelo spaCy PT + frontend)
make install

# 2. Pré-processamento (gera os splits em data/processed)
make splits

# 3a. Classificação — modelos clássicos
make train-tfidf            # TF-IDF + LR  (NGRAMS=unigrams|bigrams|both)
make champion               # BERTimbau (treino + inferência) e promove o CSV campeão

# 3b. Classificação — LLM generativo (requer o app LM Studio rodando)
make classify-llm LLM_MODEL="qwen/qwen3-4b-2507" LIMIT=50
make eval-llm   LLM_MODEL="qwen/qwen3-4b-2507"

# 4. Recomendador — construir o corpus vetorial
make build-index            # gera 04_recommender/artifacts/
make demo                   # recomendação de demonstração

# 5. Subir o site (SentiShop)
make dev                    # DEV: backend (:8000, reload) + frontend Vite (:5173)
make webapp                 # PROD: compila o frontend e serve tudo na porta 8000

# 6. Testes / validação (não-destrutivo)
make test                   # testes do recomendador
make smoke                  # valida a Makefile sem sobrescrever arquivos
```

Variáveis úteis (sobrescrevíveis): `PY` (interpretador Python), `NGRAMS`, `DATASET`,
`LLM_MODEL`, `LIMIT`, `OUT`, `PORT`. Ex.: `make install PY=/caminho/do/env/python`.

---

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/galvaowesley/sentiment-as-relevance-recommender.git
cd sentiment-as-relevance-recommender
```

### 2. Criar e ativar o ambiente

```bash
# Com mamba/conda (Python >= 3.11; o deploy usa 3.11)
mamba create -n nlp_project python=3.11
mamba activate nlp_project
```

### 3. Instalar dependências

```bash
make install        # pip install -r requirements.txt + spaCy pt_core_news_sm + npm ci
# ou manualmente:
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

---

## Branches

| Branch | Descrição |
|---|---|
| `main` | Código base, pré-processamento e estrutura geral |
| `eda` | Análise exploratória do corpus B2W-Reviews01 |
| `sentiment_classifier` | Pipeline 1 — classificadores clássicos (TF-IDF LR, BERTimbau) |
| `sentiment_classifier_llm` | Pipeline 1 — classificador generativo one-shot (LM Studio) |
| `recommender` | Pipeline 2 — motor de recomendação orientado por sentimento |

```bash
# Acessar uma branch específica
git checkout eda
git checkout sentiment_classifier
git checkout recommender
```

---

## Dados

O corpus **B2W-Reviews01** (~50MB) não está versionado no repositório. Para obtê-lo:

```bash
python -c "from datasets import load_dataset; load_dataset('ruanchaves/b2w-reviews01')"
```

Ou acesse: https://github.com/americanas-tech/b2w-reviews01

Coloque o arquivo em `data/raw/B2W-Reviews01.csv`.

---

## Pipelines

### Pipeline 1 — Classificação de Sentimento

```
Corpus B2W → Mapeamento de Polaridade → EDA → Pré-processamento
→ Representação (TF-IDF / BERTimbau / LLM) → Treinamento / Inferência
→ Avaliação (F1-macro, AUC-ROC) → Modelo Campeão
```

### Pipeline 2 — Recomendação Orientada por Sentimento

```
Modelo Campeão → Score S(p) por Produto → Embeddings (Qwen3-Embedding-0.6B)
→ Banco Vetorial (zvec/HNSW) → Busca ANN → Re-ranqueamento → Avaliação
```

---

## Score de re-ranqueamento

```
Score_final(p) = α · sim(q, p) + (1 − α) · S(p)
```

- `sim(q, p)`: similaridade de cosseno entre embeddings de título
- `S(p)`: proporção de avaliações positivas do produto (rótulos do modelo campeão)
- `α`: hiperparâmetro (default `0.7` em `04_recommender/config.py`; `05_webapp/handoff.md` explora `0.5`)
