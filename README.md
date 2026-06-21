# Sentimento como Relevância: Recomendação de Produtos em E-commerce Brasileiro

Projeto da disciplina de Processamento de Linguagem Natural — PPGCC/UFSCar.

**Autores:** Cilene Renata Real · Jayme Sakae dos Reis Furuyama · Wesley Nogueira Galvão

---

## Sobre o projeto

Sistema integrado de duas etapas para recomendação de produtos em e-commerce em português brasileiro:

1. **Classificador de sentimento** — compara TF-IDF + Regressão Logística com BERTimbau (feature extraction e fine-tuning) sobre o corpus B2W-Reviews01
2. **Motor de recomendação** — recupera produtos semanticamente similares pelo título (Serafim PT* + busca vetorial ANN) e re-ranqueia os resultados pelo score de positividade derivado do modelo campeão

---

## Estrutura do repositório

```
sentiment-as-relevance-recommender/
│
├── 01_eda/                    # Análise exploratória do corpus B2W-Reviews01
│   ├── notebooks/             # Notebooks Jupyter de exploração
│   └── figures/               # Visualizações geradas
│
├── 02_preprocessing/          # Pré-processamento textual
│   ├── common/                # Normalização compartilhada (todos os modelos)
│   └── tfidf/                 # Etapas extras: stopwords, lematização, n-gramas
│
├── 03_sentiment_classifier/   # Pipeline 1: classificação de sentimento
│   ├── tfidf_lr/              # TF-IDF + Regressão Logística (baseline)
│   ├── bertimbau_lr/          # BERTimbau [CLS] congelado + LR
│   ├── bertimbau_ft/          # BERTimbau fine-tuning + MLP
│   └── checkpoints/           # Pesos e artefatos dos modelos treinados
│
├── 04_recommender/            # Pipeline 2: recomendação orientada por sentimento
│   ├── vectorizer/            # Embeddings de títulos com Serafim PT*
│   ├── vector_store/          # Indexação e busca ANN (FAISS/ChromaDB/Qdrant)
│   └── reranking/             # Re-ranqueamento ponderado por sentimento
│
├── 05_webapp/                 # Demo: site de e-commerce simulado
│   ├── backend/               # API REST (FastAPI)
│   └── frontend/              # Interface web
│
├── 06_evaluation/             # Avaliação quantitativa e qualitativa
│   ├── classification/        # F1-macro, AUC-ROC, matriz de confusão
│   ├── recommendation/        # NDCG@k, Precision@k, Recall@k
│   └── qualitative/           # Avaliação manual e Kappa de Cohen
│
├── data/
│   ├── raw/                   # B2W-Reviews01.csv (não versionado — ver instruções)
│   ├── processed/             # Splits treino/val/teste e scores de sentimento
│   └── samples/               # Amostras pequenas para testes rápidos
│
└── docs/
    ├── reports/               # Relatórios parcial e final
    └── presentations/         # Apresentações dos seminários
```

> Cada pasta contém um `README.md` descrevendo seu propósito e conteúdo esperado.

---

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/galvaowesley/sentiment-as-relevance-recommender.git
cd sentiment-as-relevance-recommender
```

### 2. Criar e ativar o ambiente

```bash
# Com mamba/conda
mamba create -n nlp_project python=3.13
mamba activate nlp_project
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## Branches

| Branch | Descrição |
|---|---|
| `main` | Código base, pré-processamento e estrutura geral |
| `eda` | Análise exploratória do corpus B2W-Reviews01 |
| `sentiment_classifier` | Pipeline 1 — classificadores de sentimento (TF-IDF LR, BERTimbau) |
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
→ Representação (TF-IDF / BERTimbau) → Treinamento → Validação Cruzada
→ Avaliação → Modelo Campeão
```

### Pipeline 2 — Recomendação Orientada por Sentimento

```
Modelo Campeão → Score S(p) por Produto → Embeddings (Serafim PT*)
→ Banco Vetorial → Busca ANN → Re-ranqueamento → Avaliação
```

---

## Score de re-ranqueamento

```
Score_final(p) = α · sim(q, p) + (1 − α) · S(p)
```

- `sim(q, p)`: similaridade de cosseno entre embeddings de título
- `S(p)`: proporção de avaliações positivas do produto
- `α`: hiperparâmetro ajustado empiricamente
