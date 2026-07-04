# Rascunho — Pipeline 2 (Recomendação) para o relatório final

Rascunho de apoio, **não é para colar direto**: revisar, cortar e integrar manualmente
em `relatorio_final.tex`. Cada bloco indica o alvo exato (seção/subseção/linha atual).

Ordem de prioridade pedida: (1) atualizar a metodologia já escrita para refletir o que
foi de fato implementado, (2) escrever os resultados qualitativos do Pipeline 2 na seção
de Discussão, (3) por último, a proposta de avaliação objetiva (NDCG@k / Precision@k /
Recall@k) com critério de relevância por rating normalizado — **essa parte ainda não
tem números calculados**, é só o desenho do experimento.

---

## Bloco 0 — Parágrafo sobre o webapp (sem detalhes técnicos)

**Alvo:** um parágrafo curto de contexto, útil tanto na abertura de
`\subsection{Visão Geral dos Pipelines}` (linha 260) quanto na abertura de
`\subsection{Pipeline 2 -- Recomendação Orientada por Sentimento}` (linha 904) —
escolham o encaixe conforme o restante do texto ao redor.

```latex
Para tornar o Pipeline 2 tangível, o projeto inclui uma aplicação web (SentiShop)
que simula a experiência de navegação de um site de e-commerce brasileiro.\footnote{Disponível em \url{https://sentiment-as-relevance-recommender-production.up.railway.app}.}
Nela é possível percorrer o catálogo por categoria, buscar produtos por nome e
abrir a página de um produto específico com suas avaliações. Ao acessar um
produto, a aplicação exibe um painel de recomendações com os itens mais
similares sugeridos pelo sistema, permitindo ajustar interativamente o peso
entre similaridade semântica e sentimento na composição do ranqueamento e
observar, na prática, como essa escolha altera a lista de produtos sugeridos. A
aplicação serviu tanto como demonstração do funcionamento do motor de
recomendação quanto como ferramenta de apoio à avaliação qualitativa
(Seção~\ref{subsec:qual_recommendation}).
```

---

## Bloco 1 — Atualizar `\subsection{Pipeline 2 -- Recomendação Orientada por Sentimento}`

**Alvo:** `relatorio_final.tex`, linhas 904–968 (`\label{subsec:pipeline2}`).

O texto atual está no futuro/condicional ("será utilizado", "candidatos como FAISS,
ChromaDB e Qdrant, cuja escolha será definida..."), porque foi escrito como proposta.
O código em `04_recommender/` já resolveu essas decisões — o texto deve virar
descrição do que **foi** feito, não do que se planejava fazer. Principais divergências
achado vs. texto atual:

| Item | Texto atual (proposta) | Implementado |
|---|---|---|
| Modelo de embedding | Serafim PT* (`serafim-335m-portuguese-pt-sentence-encoder`) | **Qwen3-Embedding-0.6B**, 1024-d, L2-normalizado |
| Banco vetorial | "candidatos como FAISS, ChromaDB e Qdrant" (em aberto) | **zvec**, índice **HNSW**, métrica cosseno |
| Corte de avaliações mínimas | um único `n_min=5` para "inclusão no índice" | **dois cortes distintos**: `min_reviews=1` (vetorização/índice) e `recommend_min_reviews=5` (elegibilidade para ser recomendado, aplicado em query-time) |
| $K$ candidatos | "definido empiricamente" | `retrieve_k=50` |
| $\alpha$ | hiperparâmetro livre | `alpha=0.7` (default de produção) |
| Encoding assimétrico query/doc | não mencionado | títulos do corpus como *documento* (sem instrução), produto de consulta como *query* (com prefixo de instrução) — recomendado pelo Qwen3-Embedding |

### 1.1 `\subsubsection{Score de Sentimento por Produto}` (linha 907)

Sugestão de ajuste (mantém a Eq. \ref{eq:score}, só ajusta o texto ao redor):

```latex
O modelo campeão da etapa de classificação (BERTimbau + Regressão Logística,
Seção~\ref{subsec:pipeline1}) foi aplicado a todas as avaliações do corpus (splits
de validação e teste, sem uso de dados de treino) para classificá-las como
positivas ou negativas. Para cada produto único (identificado pelo
\texttt{product\_id}), calcula-se um \textit{score} de relevância $S(p)$ definido
como:

\begin{equation}
  S(p) = \frac{|\{\text{avaliações positivas de } p\}|}{|\{\text{todas as avaliações de } p\}|}
  \label{eq:score}
\end{equation}

$S(p)$ é obtido diretamente da coluna \texttt{inferencia\_bertimbau} do CSV de
inferência (protocolo \texttt{SentimentScorer}, com a implementação de produção
\texttt{PredictedLabelSentimentScorer}), de modo que o mesmo score real alimenta
tanto o índice de recomendação (corpus) quanto a vitrine simulada (storefront) —
ambos derivados da mesma fonte, eliminando qualquer divergência entre o que é
indexado e o que é exibido ao usuário.
```

### 1.2 `\subsubsection{Vetorização dos Títulos de Produtos}` (linha 919)

```latex
Os títulos de produto (\texttt{product\_name}) são vetorizados com o
\textbf{Qwen3-Embedding-0.6B}, um \textit{sentence encoder} multilíngue que produz
vetores densos de 1024 dimensões, normalizados em norma L2 (o que torna o produto
escalar equivalente à similaridade de cosseno). Segue-se o esquema assimétrico
recomendado para a família Qwen3-Embedding: títulos do corpus são codificados como
\textit{documentos} (sem prefixo), enquanto o produto de consulta é codificado como
\textit{query}, com o prefixo de instrução \textit{"Given a retail product title,
retrieve titles of similar products"}. Todos os produtos do catálogo são
vetorizados (\texttt{min\_reviews=1}); o corte por número mínimo de avaliações é
aplicado apenas no momento da recomendação (Seção~\ref{subsubsec:vectordb}), não na
indexação, preservando o catálogo completo como espaço de busca.
```

*(Nota: se quiserem manter Serafim como trabalho relacionado/alternativa cogitada, dá
pra citar em "Trabalhos Relacionados" — mas na metodologia realizada o modelo é o
Qwen3.)*

### 1.3 `\subsubsection{Banco Vetorial e Recuperação}` (linha 923)

```latex
Os vetores de título foram indexados no \textbf{zvec}, um banco vetorial
\textit{open-source} embarcado, em uma coleção HNSW (\textit{Hierarchical Navigable
Small World}) com cosseno como métrica de distância. A busca por similaridade é
feita por \textit{Approximate Nearest Neighbors} (ANN):

\begin{equation}
  \text{sim}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{|\vec{u}|\, |\vec{v}|}
  \label{eq:cosine}
\end{equation}

Dado um produto de consulta $q$ (vetorizado como \textit{query}), o sistema
recupera os $K=50$ títulos mais similares do índice (\texttt{retrieve\_k}) — um
\textbf{pool bruto} por similaridade, maior que o número final de recomendações
exibidas. O filtro de elegibilidade — produto com ao menos
\texttt{recommend\_min\_reviews}$=5$ avaliações — é aplicado neste momento,
diretamente na consulta ANN (com uma guarda adicional em Python contra o caso de o
backend ignorar o filtro), e não na construção do índice: o índice mantém o
catálogo completo (17.081 produtos vetorizados), e apenas a recomendação é
restrita aos produtos com suporte estatístico suficiente em $S(p)$. Desses até 50
candidatos, o re-ranqueamento por sentimento (Seção~\ref{subsubsec:reranking})
reordena e corta para o \texttt{top\_n}$=10$ efetivamente exibido ao usuário — os
dois números têm papéis distintos: $K$ controla o tamanho do pool candidato à
reordenação, \texttt{top\_n} controla quantos aparecem na vitrine.
```

### 1.4 `\subsubsection{Re-ranqueamento por Sentimento}` (linha 935)

Pode ficar quase como está — só trocar "hiperparâmetro" por "hiperparâmetro fixado
empiricamente em $\alpha=0.7$ na configuração de produção" e citar `Score_final`
igual à Eq. \ref{eq:reranking} já presente.

### 1.5 `\subsubsection{Métricas de Avaliação -- Recomendação}` (linha 947)

Manter como está para a parte descritiva de Precision@k/Recall@k/NDCG@k — essas
fórmulas continuam corretas. Só sugiro **não travar** ainda o critério de
relevância ($S(p)>0.7$ / $S(p)<0.3$) como único: no Bloco 3 deste rascunho proponho
um segundo critério (rating normalizado por categoria) para a avaliação objetiva
por categoria, que é conceitualmente mais limpo por não reciclar o próprio termo
usado no re-ranqueamento (ver justificativa no Bloco 3). Se quiserem, adicionem uma
frase de transição tipo:

```latex
Além do critério baseado em $S(p)$ — adequado para auditar o comportamento do
re-ranqueamento em si — a Seção~\ref{subsec:pipeline2_avaliacao_objetiva} propõe um
segundo critério de relevância, independente da fórmula de re-ranqueamento, baseado
no rating médio do produto normalizado dentro de sua categoria.
```

### 1.6 `\subsubsection{Avaliação Qualitativa -- Recomendação}` (linha 965)

Aqui o texto atual descreve um plano ("cada integrante... avaliará... 20 pares").
O que foi **de fato executado** e está em `06_evaluation/recommendation/` é uma
avaliação qualitativa dirigida por categoria (não por amostragem aleatória de 20
pares com múltiplos avaliadores/Kappa). Sugiro decidir: (a) se a avaliação com
Kappa por 20 pares foi mesmo feita por outro integrante do grupo — mantém o texto;
(b) se não foi feita, substituir pela descrição real (ver Bloco 2) e mover a
menção ao Kappa para "trabalhos futuros" ou remover.

---

## Bloco 2 — Nova seção de Resultados do Pipeline 2

**Alvo:** dentro de `\section{Discussão dos Resultados}` (linha 972), como nova
`\subsection` — sugiro logo após `\subsection{Eficiência Computacional...}` (linha
1046) e antes de `\section{Considerações Finais}` (linha 1062), já que hoje toda a
seção de Discussão só cobre o Pipeline 1 (classificação). Proposta de label:
`\label{subsec:resultados_pipeline2}`.

Base: `06_evaluation/recommendation/recommendation_eval.json` +
`04_recommender/artifacts/meta.json` (índice com 17.081 produtos, gerado a partir do
CSV inferido pelo BERTimbau em val+test).

```latex
\subsection{Resultados do Pipeline 2 -- Recomendação Orientada por Sentimento}
\label{subsec:resultados_pipeline2}

Para avaliar qualitativamente o motor de recomendação em produção
($\alpha=0.7$, $\text{top\_n}=10$, $\text{recommend\_min\_reviews}=5$), foram
contrastadas duas categorias do catálogo com perfis opostos de risco: uma de baixo
risco e alto volume (\textit{Celulares e Smartphones}: 1.247 produtos, 8.363
avaliações, 250 recomendáveis) e uma de alto risco (\textit{Móveis}, a categoria
com \textbf{mais} produtos do catálogo, 2.028, porém com o pior sentimento médio
entre as categorias grandes, sendo $\bar{S}$ dos recomendáveis igual a $0{,}39$,
com 23 dos 61 produtos recomendáveis apresentando $S(p) < 0{,}3$).

Em \textit{Celulares e Smartphones}, as recomendações para dois produtos de
consulta (Motorola Moto G 5S e Samsung Galaxy J7 Metal) resultaram em 10/10 itens
de alta relevância semântica ($n_{\text{off\_topic}}=0$, com similaridade média
entre 0,80 e 0,84) e de sentimento alto (score médio de sentimento entre 0,88 e
0,91), todos dentro da própria categoria. Esse comportamento confirma a hipótese
de que, em categorias grandes, coerentes e majoritariamente bem avaliadas, o
re-ranqueamento por sentimento reforça um resultado semântico já de boa qualidade,
sem custo aparente de relevância.

Em \textit{Móveis}, a mesma configuração expõe dois modos de falha distintos:

\begin{itemize}[noitemsep]
  \item \textbf{Deriva de tema por baixa similaridade}: para a consulta "Cama
  Solteiro Barcelona" ($S(p)=0{,}17$), a similaridade média dos candidatos
  recuperados foi de apenas $0{,}44$, abaixo do limiar de $0{,}5$ usado para
  marcar um item como fora de tema. Todas as 10 recomendações ficaram fora do
  tema, sendo a primeira colocada uma "Bomba Tira-Leite Materno", com
  $S(p)=1{,}0$, pois o termo de sentimento na Eq.~\ref{eq:reranking} recruta
  itens de alto $S(p)$ de categorias completamente distintas quando não há
  vizinhos semânticos de qualidade suficiente dentro da própria categoria.
  \item \textbf{Falha relevância-sentimento em quase-duplicatas}: para a consulta
  "Guarda Roupa Casal Conjugado" ($S(p)=0{,}22$), a recomendação \#1 é uma
  variante de cor do mesmo produto, com similaridade altíssima, $\approx 0{,}86$,
  mas $S(p)=0{,}20$, ou seja, um produto mal avaliado. Nesse caso, a similaridade
  domina o termo de re-ranqueamento e promove ao topo um item semanticamente
  quase idêntico, porém de baixa qualidade percebida, o que o critério de
  relevância por $S(p)$ (Seção~\ref{subsubsec:metrics_recomendation}) classificaria
  como "pouco relevante".
\end{itemize}

Esses resultados indicam que, com $\alpha=0{,}7$, o re-ranqueamento por sentimento
é robusto no sentido de raramente promover produtos de $S(p)$ baixo isoladamente,
de modo que o risco de recomendação ruim manifesta-se sobretudo como
\textbf{item fora de tema} em categorias semanticamente dispersas ou mal
cobertas pelo índice, e apenas excepcionalmente como sentimento baixo, no caso
específico de quase-duplicatas em que a similaridade extrema neutraliza o peso
$(1-\alpha)$ do termo de sentimento. Isso sugere que trabalhos futuros poderiam
considerar um piso mínimo de similaridade, rejeitando candidatos com
$\text{sim}(q,p)$ abaixo de um limiar antes de aplicar o re-ranqueamento por
sentimento, especialmente em categorias de alto risco.
```

*(Números vêm do `README.md` de `06_evaluation/recommendation/` e do
`recommendation_eval.json`; confiram os valores exatos antes de publicar, e
considerem incluir uma tabela com as colunas `mean_similarity`, `mean_sentiment`,
`n_off_topic`, `n_same_category` para os 4 pares consulta-recomendação.)*

---

## Bloco 3 — Avaliação objetiva (NDCG@k, Precision@k, Recall@k) — **fazer por último**

**Alvo:** duas frentes —

1. Metodologia: nova `\subsubsection` dentro de `subsec:pipeline2`, logo após
   `\label{subsubsec:qual_recommendation}` (linha 968), por exemplo
   `\label{subsec:pipeline2_avaliacao_objetiva}` (já referenciado no Bloco 1.5).
2. Resultados: nova `\subsection` dentro de `\section{Discussão dos Resultados}`,
   depois do Bloco 2 (`subsec:resultados_pipeline2}`), com os números calculados.

Isto ainda não tem números — é a proposta de desenho experimental para vocês
rodarem o script e depois preencherem a tabela de resultados.

### 3.1 Por que não usar $S(p)$ como critério de relevância aqui

$S(p)$ já é um dos dois termos da própria fórmula de re-ranqueamento
(Eq.~\ref{eq:reranking}). Usar $S(p)$ como rótulo de relevância para medir
Precision@k/Recall@k/NDCG@k mediria, em parte, se o sistema recuperou o que ele
mesmo foi desenhado para priorizar — um viés de circularidade. O rating médio do
produto (\texttt{overall\_rating}, 1--5, disponível por avaliação no CSV bruto e já
agregado como \texttt{avg\_rating} no backend do webapp,
\texttt{05\_webapp/backend/app.py}) é um sinal de qualidade percebida
\textbf{independente} da fórmula de ranqueamento (que usa apenas similaridade e
$S(p)$, nunca o rating bruto), por isso é a escolha mais defensável como critério
de relevância para a avaliação objetiva.

### 3.2 Definição de relevância proposta

```latex
Para a avaliação objetiva por categoria, definimos a relevância de um produto $p$
pertencente à categoria $c$ a partir de seu rating médio
$\overline{r}(p) = \frac{1}{n_p}\sum_i r_i(p)$ (média das notas
\texttt{overall\_rating}, em $\{1,\ldots,5\}$, de todas as avaliações de $p$),
normalizado \textbf{dentro da categoria} por min--max:

\begin{equation}
  \text{rel}(p) = \frac{\overline{r}(p) - \min_{p' \in c}\overline{r}(p')}
                       {\max_{p' \in c}\overline{r}(p') - \min_{p' \in c}\overline{r}(p')}
  \label{eq:relevancia_rating}
\end{equation}

restrito ao conjunto de produtos recomendáveis da categoria (i.e., com
$\text{num\_reviews} \geq 5$). $\text{rel}(p) \in [0, 1]$ é usado diretamente como
\textit{grau de relevância contínuo} no cálculo do ganho do NDCG (ganho linear
$\text{rel}(p)$ em vez do ganho exponencial $2^{r}-1$, mais adequado a rótulos
discretos). Para Precision@k e Recall@k, que exigem um rótulo binário
relevante/não relevante, um produto é considerado relevante se
$\text{rel}(p) \geq 0{,}7$ -- por simetria com o limiar já adotado para $S(p)$ na
Seção~\ref{subsubsec:metrics_recomendation}, porém aplicado ao rating normalizado
em vez do sentimento.
```

### 3.3 Desenho do experimento (categoria Smartphone)

```latex
A avaliação objetiva foi conduzida sobre a categoria \textit{Celulares e
Smartphones} (\texttt{site\_category\_lv1}), reaproveitando o mesmo produto de
consulta principal da avaliação qualitativa (Seção~\ref{subsec:resultados_pipeline2}):
"Smartphone Motorola Moto G 5S Dual Chip Android 7.1.1 Nougat Tela 5.2\"
Snapdragon 430 32GB 4G Câmera 16MP - Platinum" (\texttt{product\_id=132444092},
342 avaliações, rating médio 4,03, $\text{rel}(q)=0{,}76$ na categoria). Para essa
consulta $q$:

\begin{enumerate}[noitemsep]
  \item o motor retorna os $k$ produtos mais bem ranqueados dentre os
  candidatos recomendáveis da categoria ($\text{top\_n}=10$, $\alpha=0{,}7$);
  \item o conjunto de referência de itens relevantes é definido sobre
  \textbf{todos} os produtos recomendáveis da categoria (excluindo $q$), via
  Eq.~\ref{eq:relevancia_rating} com o limiar $\text{rel}(p)\geq0{,}7$;
  \item calculam-se Precision@k, Recall@k e NDCG@k para $k \in \{5, 10\}$.
\end{enumerate}

Como a categoria \textit{Celulares e Smartphones} tem 250 produtos recomendáveis, o
conjunto de itens relevantes tende a ser muito maior que $k$, de modo que Recall@k
é estruturalmente baixo em valor absoluto e deve ser lido de forma comparativa, por
exemplo entre diferentes valores de $\alpha$, ou entre \textit{Smartphones} e
\textit{Móveis}, e não como cobertura absoluta do universo de itens relevantes.

O experimento aqui foi executado para um único produto de consulta, como recorte
inicial de validação do método; o mesmo script se aplica diretamente ao segundo
produto de consulta da avaliação qualitativa (Samsung Galaxy J7 Metal,
\texttt{product\_id=128010777}) e à categoria \textit{Móveis}, bastando trocar os
parâmetros de entrada.
```

### 3.4 Como calcular (roteiro de implementação, não é para o texto do artigo)

Script sugerido em `06_evaluation/recommendation/objective_metrics.py`:

1. Ler o CSV bruto (`B2W-Reviews01_inferred_bertimbau.csv`) e agregar
   `avg_rating` por `product_id` — mesma lógica já usada em
   `05_webapp/backend/app.py` (`_load_reviews`, linhas ~60-90): média de
   `overall_rating` por produto.
2. Filtrar a categoria `site_category_lv1 == "Celulares e Smartphones"` e
   `num_reviews >= 5` (mesmo corte de `recommend_min_reviews`).
3. Aplicar a Eq.~\ref{eq:relevancia_rating} (min–max do `avg_rating` dentro do
   subconjunto filtrado).
4. Para cada produto de consulta, chamar `GET /recommend?product_id=...&top_n=K&alpha=0.7`
   no backend já de pé (`http://127.0.0.1:8000`), com `K` grande o bastante (ex.: 10)
   e casar cada `product_id` retornado com `rel(p)`.
5. Implementar Precision@k/Recall@k/NDCG@k (não precisa de lib extra —
   ~20 linhas de numpy) e agregar por média entre consultas.

Isso é reaproveitável para rodar a mesma avaliação em \textit{Móveis} depois, como
comparação de robustez do re-ranqueamento entre categoria de baixo e alto risco —
mesmo desenho, outro filtro de categoria.

### 3.5 Resultados (executado — 1 produto de consulta)

**Status:** já rodado. Script: `06_evaluation/recommendation/objective_metrics.py`.
Saída completa (recomendações item a item, rating, relevância):
`06_evaluation/recommendation/objective_metrics_smartphone.json`.

| Parâmetro | Valor |
|---|---|
| Produto de consulta | Smartphone Motorola Moto G 5S ... Platinum (`132444092`) |
| Categoria | Celulares e Smartphones |
| Produtos recomendáveis na categoria (≥5 avaliações) | 250 |
| Itens relevantes na categoria (rel(p) ≥ 0,7, excluindo a consulta) | 126 |
| $\alpha$ / top\_n | 0,7 / 10 |

| k | Precision@k | Recall@k | NDCG@k |
|---|---|---|---|
| 5  | 0,800 | 0,032 | 0,947 |
| 10 | 0,900 | 0,071 | 0,987 |

```latex
\subsection{Avaliação Objetiva do Pipeline 2 -- Categoria Smartphone}
\label{subsec:resultados_pipeline2_objetiva}

A avaliação objetiva foi executada para o produto de consulta "Smartphone
Motorola Moto G 5S Dual Chip Android 7.1.1 Nougat Tela 5.2\" Snapdragon 430
32GB 4G Câmera 16MP - Platinum" (\texttt{product\_id=132444092}), na categoria
\textit{Celulares e Smartphones} (250 produtos recomendáveis, dos quais 126
relevantes segundo o critério da Eq.~\ref{eq:relevancia_rating} com limiar
$0{,}7$). Os resultados são apresentados na Tabela~\ref{tab:objetiva_smartphone}.

\begin{table}[htbp]
  \centering
  \caption{Precision@k, Recall@k e NDCG@k para a consulta Smartphone Motorola
  Moto G 5S (categoria Celulares e Smartphones, $\alpha=0{,}7$).}
  \label{tab:objetiva_smartphone}
  \begin{tabular}{crrr}
    \toprule
    \textbf{k} & \textbf{Precision@k} & \textbf{Recall@k} & \textbf{NDCG@k} \\
    \midrule
    5  & 0,800 & 0,032 & 0,947 \\
    10 & 0,900 & 0,071 & 0,987 \\
    \bottomrule
  \end{tabular}
\end{table}

Os valores de Precision@k e NDCG@k são altos (0,80 e 0,90 em Precision@5 e
Precision@10; 0,95 e 0,99 em NDCG@5 e NDCG@10), confirmando quantitativamente o
que a avaliação qualitativa já indicava para essa categoria: os itens
recuperados por similaridade semântica tendem a ser também bem avaliados pelos
consumidores (rating médio alto), e o re-ranqueamento por sentimento não
degrada essa relevância, colocando os poucos itens de rating mais baixo em
posições finais da lista. O único item não relevante em Precision@5 é o
\texttt{product\_id=132444076} (mesma linha "Moto G 5S", cor Azul Safira, com
rating médio de $3{,}6$ e $\text{rel}(p)=0{,}65$, logo abaixo do limiar de
$0{,}7$), promovido pela alta similaridade textual ($0{,}894$) mesmo com
sentimento moderado ($S(p)=0{,}80$).

Já o Recall@k é baixo em valor absoluto (0,032 em $k=5$; 0,071 em $k=10$), o que
é esperado dado o desenho do experimento: com 126 produtos relevantes na
categoria e apenas $k \leq 10$ posições na lista, nenhum sistema de recomendação
razoável atingiria recall alto nesse regime. O valor é mais útil como referência
comparativa (por exemplo, para outros valores de $\alpha$ ou para a categoria
\textit{Móveis}) do que como medida de cobertura absoluta.

Vale notar que este resultado foi obtido para um único produto de consulta, como
prova de conceito do método objetivo de avaliação; a extensão para o segundo
produto de consulta da avaliação qualitativa e para a categoria \textit{Móveis}
é direta, bastando reexecutar o script com outros parâmetros.
```
