# Makefile — Sentimento como Relevância: Recomendação de Produtos em E-commerce Brasileiro
#
# Ponto de entrada único para o projeto. Rode `make help` para ver todos os alvos.
# Todas as variáveis abaixo podem ser sobrescritas na linha de comando, ex.:
#   make install PY=/caminho/do/env/python
#   make classify-llm LLM_MODEL="qwen/qwen3-4b-2507" LIMIT=50
#   make build-index OUT=04_recommender/artifacts_demo LIMIT=2000

# ----------------------------------------------------------------------------
# Variáveis (sobrescrevíveis com make VAR=valor)
# ----------------------------------------------------------------------------
# NOTA: comentários ficam em linhas próprias — comentário inline após `?=`
# deixa espaços em branco grudados no valor da variável (quebra nomes citados).

# Interpretadores / ferramentas
PY         ?= python
PIP        ?= $(PY) -m pip
NPM        ?= npm
ROOT       := $(shell pwd)

# Caminhos de projeto
FRONTEND_DIR := 05_webapp/frontend
BACKEND_DIR  := 05_webapp/backend
REC_DIR      := 04_recommender

# Dados / classificação
RAW_CSV    ?= data/raw/B2W-Reviews01.csv
# NGRAMS: unigrams | bigrams | both
NGRAMS     ?= both
# DATASET: vazio = com neutros; "no_neutral" = sem neutros
DATASET    ?=
SPLITS_OUT ?= data/processed

# LLM (LM Studio) — LLM_MODEL é o id do modelo; liste com `lms ls`
LLM_MODEL  ?= qwen/qwen3-4b-2507
LLM_PRED   ?= 03_sentiment_classifier/checkpoints/llm_$(subst :,_,$(subst /,_,$(LLM_MODEL)))_predictions.csv

# Recomendador — OUT/ARTIFACTS permitem redirecionar a saída p/ dir descartável
OUT        ?= 04_recommender/artifacts
ARTIFACTS  ?=
PRODUCT_ID ?=

# Webapp / rede
PORT       ?= 8000
SMOKE_PORT ?= 8123

# Opcional: --limit N para smoke tests
LIMIT      ?=

.DEFAULT_GOAL := help

# ----------------------------------------------------------------------------
##@ Ajuda
# ----------------------------------------------------------------------------

help: ## Mostra esta ajuda
	@awk 'BEGIN {FS = ":.*##"; printf "\nSentiShop — alvos disponíveis\n  uso: \033[36mmake <alvo>\033[0m\n"} \
		/^[a-zA-Z0-9_.-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

# ----------------------------------------------------------------------------
##@ Instalação
# ----------------------------------------------------------------------------

install: install-py install-spacy install-web ## Instala TODAS as dependências (Python + spaCy PT + frontend)
	@echo "==> Instalação concluída."

install-py: ## Instala as dependências Python (requirements.txt) no ambiente ativo
	$(PIP) install -r requirements.txt

install-spacy: ## Baixa o modelo spaCy pt_core_news_sm (obrigatório p/ TF-IDF)
	$(PY) -m spacy download pt_core_news_sm

install-web: ## Instala as dependências do frontend (npm ci)
	cd $(FRONTEND_DIR) && $(NPM) ci

# ----------------------------------------------------------------------------
##@ Dados & pré-processamento
# ----------------------------------------------------------------------------

data: ## (opcional) Baixa o corpus B2W-Reviews01 via HuggingFace (best-effort)
	@echo "==> Baixando B2W-Reviews01 (cache HuggingFace)..."
	$(PY) -c "from datasets import load_dataset; load_dataset('ruanchaves/b2w-reviews01')"
	@echo "IMPORTANTE: exporte/copie o CSV para $(RAW_CSV) se ainda não existir."

splits: ## Gera os splits treino/val/teste (com e sem neutros) em $(SPLITS_OUT)
	$(PY) 02_preprocessing/split_data.py --input $(RAW_CSV) --output-dir $(SPLITS_OUT)
	$(PY) 02_preprocessing/split_data_no_neutral.py --input $(RAW_CSV) --output-dir $(SPLITS_OUT)

# ----------------------------------------------------------------------------
##@ Classificação — modelos clássicos
# ----------------------------------------------------------------------------

train-tfidf: ## Treina TF-IDF + Regressão Logística (NGRAMS=unigrams|bigrams|both)
	$(PY) 03_sentiment_classifier/tfidf_lr/train_tfidf.py --ngrams $(NGRAMS)

infer-tfidf: ## Inferência TF-IDF+LR -> CSV enriquecido (DATASET vazio=com neutros)
	$(PY) 03_sentiment_classifier/tfidf_lr/eval_tfidf.py --ngrams $(NGRAMS) --dataset "$(DATASET)"

train-bertimbau: ## Treina BERTimbau [CLS] congelado + Regressão Logística
	$(PY) 03_sentiment_classifier/bertimbau_lr/train_bertimbau_lr.py

infer-bertimbau: ## Inferência BERTimbau+LR -> CSV enriquecido (modelo campeão)
	$(PY) 03_sentiment_classifier/bertimbau_lr/eval_bertimbau_lr.py --dataset "$(DATASET)"

classify-classic: train-tfidf infer-tfidf train-bertimbau infer-bertimbau ## Fluxo clássico completo (TF-IDF + BERTimbau, treino + inferência)
	@echo "==> Fluxo clássico concluído."

champion: train-bertimbau infer-bertimbau ## Treina campeão e promove o CSV inferido p/ o recomendador
	@mkdir -p data/results_from_best_model
	cp 03_sentiment_classifier/inference/outputs/B2W-Reviews01_inferred_bertimbau.csv data/results_from_best_model/
	@echo "==> CSV campeão copiado para data/results_from_best_model/"

# ----------------------------------------------------------------------------
##@ Classificação — LLM generativo (LM Studio)
# ----------------------------------------------------------------------------

classify-llm: ## Inferência one-shot via LM Studio (LLM_MODEL=... ; use LIMIT=N p/ smoke)
	$(PY) 03_sentiment_classifier/llm/infer_llm.py --model "$(LLM_MODEL)" $(if $(LIMIT),--limit $(LIMIT),)

eval-llm: ## Recalcula métricas (F1-macro/AUC) do CSV de predições do LLM
	$(PY) 03_sentiment_classifier/llm/evaluate_llm.py --predictions "$(LLM_PRED)" --model "$(LLM_MODEL)"

# ----------------------------------------------------------------------------
##@ Recomendador (corpus vetorial)
# ----------------------------------------------------------------------------

build-index: ## Constrói o índice vetorial zvec + catálogos em $(OUT) (use LIMIT=N p/ smoke)
	cd $(REC_DIR) && $(PY) build_index.py --out $(patsubst $(REC_DIR)/%,%,$(OUT)) $(if $(LIMIT),--limit $(LIMIT),)

demo: ## Recomendação de demonstração p/ um produto (ARTIFACTS=... PRODUCT_ID=...)
	cd $(REC_DIR) && $(PY) demo.py $(if $(ARTIFACTS),--artifacts $(patsubst $(REC_DIR)/%,%,$(ARTIFACTS)),) $(if $(PRODUCT_ID),--product-id $(PRODUCT_ID),)

# ----------------------------------------------------------------------------
##@ Avaliação & testes
# ----------------------------------------------------------------------------

test: ## Testes do recomendador (rápidos, sem download de modelo)
	cd $(REC_DIR) && $(PY) -m pytest

test-integration: ## Testes de integração (baixa o modelo Qwen3-0.6B real)
	cd $(REC_DIR) && $(PY) -m pytest -m integration

eval-recommender: ## Métricas de ranqueamento (P@k/Recall@k/NDCG@k) — exige a API no ar (make backend)
	$(PY) 06_evaluation/recommendation/objective_metrics.py

# ----------------------------------------------------------------------------
##@ Webapp (SentiShop)
# ----------------------------------------------------------------------------

backend: ## Sobe só a API FastAPI (:$(PORT), com --reload)
	PYTHONPATH=$(ROOT)/$(REC_DIR) $(PY) -m uvicorn app:app --app-dir $(BACKEND_DIR) --port $(PORT) --reload

frontend: ## Sobe só o frontend Vite em modo dev (:5173)
	cd $(FRONTEND_DIR) && $(NPM) run dev

dev: ## DEV: backend (:$(PORT), reload) + frontend Vite (:5173) em paralelo
	@echo "Backend :$(PORT) (reload) | Frontend :5173 — Ctrl-C encerra ambos"
	@trap 'kill 0' INT TERM EXIT; \
		PYTHONPATH=$(ROOT)/$(REC_DIR) $(PY) -m uvicorn app:app --app-dir $(BACKEND_DIR) --port $(PORT) --reload & \
		( cd $(FRONTEND_DIR) && $(NPM) run dev ) & \
		wait

build-web: ## Compila o frontend para produção ($(FRONTEND_DIR)/dist)
	cd $(FRONTEND_DIR) && $(NPM) run build

webapp: build-web ## PROD: compila o frontend e serve tudo pela API numa porta única (:$(PORT))
	PYTHONPATH=$(ROOT)/$(REC_DIR) $(PY) -m uvicorn app:app --app-dir $(BACKEND_DIR) --host 0.0.0.0 --port $(PORT)

# ----------------------------------------------------------------------------
##@ Validação, Docker & utilidades
# ----------------------------------------------------------------------------

smoke: ## Valida a Makefile de ponta a ponta SEM sobrescrever arquivos commitados
	@echo "==> Versões"
	@make --version | head -1
	@$(PY) --version
	@$(NPM) --version
	@$(PY) -m pytest --version
	@echo "==> pytest do recomendador (tmp_path, sem downloads)"
	@$(MAKE) test
	@echo "==> demo contra cópia descartável $(REC_DIR)/artifacts_demo/"
	@rm -rf $(REC_DIR)/artifacts_demo
	@cp -r $(REC_DIR)/artifacts $(REC_DIR)/artifacts_demo
	@$(MAKE) demo ARTIFACTS=$(REC_DIR)/artifacts_demo
	@echo "==> backend /health contra artifacts_demo/ (porta $(SMOKE_PORT))"
	@RECOMMENDER_ARTIFACTS=$(REC_DIR)/artifacts_demo PYTHONPATH=$(ROOT)/$(REC_DIR) \
		$(PY) -m uvicorn app:app --app-dir $(BACKEND_DIR) --port $(SMOKE_PORT) & \
		SERVER_PID=$$!; \
		for i in $$(seq 1 40); do curl -sf localhost:$(SMOKE_PORT)/health >/dev/null 2>&1 && break || sleep 1; done; \
		curl -sf localhost:$(SMOKE_PORT)/health && echo " -> /health OK" || (echo "FALHOU"; kill $$SERVER_PID 2>/dev/null; exit 1); \
		kill $$SERVER_PID 2>/dev/null || true
	@echo "==> Dry-run dos alvos que escrevem/são pesados"
	@$(MAKE) -n splits champion classify-classic classify-llm build-index infer-tfidf infer-bertimbau eval-llm webapp dev >/dev/null && echo "dry-run OK"
	@rm -rf $(REC_DIR)/artifacts_demo
	@echo "==> smoke concluído (nenhum arquivo commitado alterado)."

docker-build: ## Constrói a imagem Docker (backend + frontend compilado)
	docker build -t sentishop .

docker-run: ## Roda a imagem Docker publicando a porta $(PORT)
	docker run --rm -p $(PORT):8000 sentishop

clean: ## Remove caches (__pycache__, .pytest_cache, artifacts_demo) — não toca em dados/artefatos versionados
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(REC_DIR)/artifacts_demo
	@echo "==> Limpeza concluída."

.PHONY: help install install-py install-spacy install-web data splits \
	train-tfidf infer-tfidf train-bertimbau infer-bertimbau classify-classic champion \
	classify-llm eval-llm build-index demo test test-integration eval-recommender \
	backend frontend dev build-web webapp smoke docker-build docker-run clean
