# Backlog

Itens em aberto identificados ao longo do desenvolvimento, organizados por
área. Nenhum bloqueia o uso atual do pacote — são melhorias, validações e
extensões para priorizar depois.

## Validação científica (alta prioridade — afeta a confiabilidade dos números)

- [ ] **Resolver `membership_side`**: comparar `"target"` (segue a Eq. 6 do
  artigo) vs. `"source"` (reproduz o comportamento do C original) nos
  datasets do paper e ver qual bate com a Tabela I publicada. Hoje os dois
  existem como opção, mas nenhum foi validado contra os números originais.
- [x] **Confirmar com uma segunda seed** se o Random search realmente perde
  de GA/PSO no Thyroid de forma consistente — **resolvido, com resposta
  diferente da esperada**: não perde de forma consistente. Após corrigir o
  bug de metodologia (GA/PSO rodando com 1 iteração só, quase um sorteio) e
  medir de novo numa máquina Linux dedicada (sem o ruído de tempo do
  Windows), uma varredura exaustiva de `sigma` (k_max=20 fixo, 11 pontos de
  0.2 a 1.2) revelou que **o espaço de busca é multimodal**: dois platôs de
  acurácia (~0.74 em sigma 0.4-0.6, ~0.75-0.76 em sigma 1.1-1.2) separados
  por um vale (~0.73 em sigma 0.7-0.9). GA e PSO convergiram para
  sigma≈0.92 -- bem no meio do vale, um ótimo local -- enquanto o Random,
  sorteando sem viés de vizinhança, encontrou o platô melhor por acidente.
  Resultado salvo em `results/thyroid/sigma_sweep_20260915T151530Z.csv`.
- [ ] **NOVO, motivado pelo achado acima**: GA/PSO ficando presos num
  ótimo local no Thyroid sugere que a população pequena (3 agentes) e o
  orçamento baixo usados nos testes não dão diversidade suficiente para
  escapar de vales no espaço de busca. Testar com população maior e/ou
  múltiplos reinícios independentes (restart strategy) antes de descartar
  GA/PSO como inadequados para esse dataset.
- [x] **Investigar o custo por avaliação do GA** ser ~2x o do PSO/Random no
  Thyroid — **resolvido**: era ruído da máquina Windows usada nos testes
  originais (143,6s vs. 83,3-83,5s ali). Confirmado ao rodar de novo numa
  máquina Linux dedicada: os três métodos deram tempos por avaliação
  consistentes entre si (71-75s). Não é uma característica real do
  algoritmo.
- [ ] **Rodar nos datasets restantes do artigo** — só testamos Boat,
  Cone-Torus e Thyroid; faltam Four-Class, Data1-5, Breast Tissue, Landsat
  Satellite, MPEG-7 BAS (dos ~12 usados no paper original).
- [ ] **Endereçar o desbalanceamento de classes** no Thyroid (achado real:
  classe minoritária "1" com recall de só 34%) — nenhuma técnica de
  correção (custo-sensível, oversampling, etc.) foi implementada ainda.
- [x] **Split estratificado**: `opfython.stream.splitter.split()` não
  estratifica por classe — pode importar em datasets desbalanceados como o
  Thyroid. **Resolvido**: `fuzzy_opf.datasets.stratified_split()`
  implementado e testado, disponível via `stratified: true` nas configs.

## Performance

- [ ] **Paralelismo** — maior limitação herdada do C (o próprio artigo cita
  isso como trabalho futuro nunca feito). O treino supervisionado é O(n²)
  sequencial; candidatos: paralelizar a avaliação de agentes do GA/PSO via
  `joblib`, ou paralelizar o cálculo de distâncias par-a-par.
- [x] **Compartilhar cache de clustering entre métodos** em
  `run_hyperparam_search.py` — hoje GA, PSO e Random cada um recalcula o
  clustering uma vez (redundante quando `k_max` é fixo entre os três).
  **Resolvido**: `cluster_cache` compartilhado, testado (confirmado que
  um `k_max` calculado por um método é reaproveitado pelos outros dois).

## Novas funcionalidades / comparações

- [ ] **Otimização Bayesiana via Optuna** — candidato mais forte pra
  "modelo baseado em distribuição" desde que o CEM quebrou; nunca
  implementado (`bayesian_search`).
- [ ] **Otimização multiobjetivo (NSGA-II/III)** — Pareto front
  acurácia-vs-custo computacional, usando `opytimizer.optimizers.multi_objective`.
  Conceitualmente separado da comparação single-objective atual.
- [ ] **CEM**: revisitar quando a `opytimizer` corrigir o bug de
  compatibilidade com NumPy 2.x (`cem_search` já existe, documentado como
  quebrado).
- [ ] **Funções de pertinência alternativas** (gaussiana, sigmoide,
  exponencial) — a Eq. 5 do artigo usa uma forma quadrática fixa; comparar
  com outras formas é uma extensão natural.
- [x] **Comparação sistemática de métricas de distância** (manhattan,
  chi_squared, bray_curtis, etc.) além da `log_squared_euclidean` padrão.
  **Resolvido**: `experiments/compare_distances.py` implementado e
  testado no Cone-Torus (mostrou diferença real: cosseno bem pior que as
  demais nesse dataset).
- [ ] **Ensemble de Fuzzy-OPF** — treinar vários modelos com
  `(k_max, sigma)` diferentes (ex.: top-5 do histórico do GA) e combinar
  por votação.

## Infraestrutura / publicação

- [ ] **Publicar no PyPI** — hoje só instala via clone + `pip install -e .`.
- [ ] **Propor PR para a própria `opfython`** — o Fuzzy-OPF preencheria uma
  lacuna real na lib (que só tem Supervised/Unsupervised/KNN/Semi-Supervised
  OPF).

## Itens menores

- [x] **Erro de rotação de log no Windows** (`WinError 32`, aparece
  repetidamente nos runs longos) — não trava a execução, mas polui a saída;
  vale suprimir ou reconfigurar o logging dos scripts de experimento.
  **Resolvido**: `fuzzy_opf/__init__.py` desativa o file handler
  problemático da `opfython` antes de qualquer outro import.
- [x] **Teste de cobertura para dataset de uma única classe** — o
  fallback de `_find_prototypes` pra esse caso nunca foi testado
  explicitamente em `test_model.py`. **Resolvido**, teste adicionado.
