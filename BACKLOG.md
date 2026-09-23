# Backlog

Itens em aberto identificados ao longo do desenvolvimento, organizados por
área. Nenhum bloqueia o uso atual do pacote — são melhorias, validações e
extensões para priorizar depois.

## Validação científica (alta prioridade — afeta a confiabilidade dos números)

- [x] **Resolver `membership_side`**: comparar `"target"` (segue a Eq. 6 do
  artigo) vs. `"source"` (reproduz o comportamento do C original) nos
  datasets do paper e ver qual bate com a Tabela I publicada. **Resolvido:
  `"target"` é a escolha correta.** Testado em Boat e Cone-Torus (20 runs
  cada, protocolo idêntico exceto `membership_side`): no Boat os dois
  modos empatam (dataset fácil demais, ~99% de acurácia, sem espaço para
  a diferença aparecer); no Cone-Torus (mais difícil, ~85%), só `"target"`
  reproduz o padrão qualitativo do artigo (Fuzzy-OPF > OPF, +0.0020 vs.
  +0.0074 no artigo) -- `"source"` empata exatamente com o OPF padrão, sem
  vantagem nenhuma. `"target"` passa a ser o padrão recomendado; `"source"`
  fica disponível só para quem quiser reproduzir o comportamento do C.
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
- [x] **GA/PSO ficando presos num ótimo local no Thyroid** — **resolvido,
  hipótese confirmada**. Com `n_agents=10` (vs. o padrão de 3) e
  `budget=40`, GA e PSO passaram a convergir para `sigma≈1.11-1.13` --
  exatamente o platô melhor identificado no sweep -- empatando com o
  Random (que já achava essa região por sorte, mesmo com população
  pequena). Confirma: o problema nunca foi GA/PSO serem inadequados para
  o Thyroid, era população pequena demais (3 agentes) sem diversidade
  para escapar do vale entre os dois platôs. `run_hyperparam_search.py`
  agora aceita `n_agents` configurável na YAML para permitir esse tipo de
  teste. Resultado em
  `results/thyroid/hyperparam_search_20260917T193701Z.csv`.
- [x] **Investigar o custo por avaliação do GA** ser ~2x o do PSO/Random no
  Thyroid — **resolvido**: era ruído da máquina Windows usada nos testes
  originais (143,6s vs. 83,3-83,5s ali). Confirmado ao rodar de novo numa
  máquina Linux dedicada: os três métodos deram tempos por avaliação
  consistentes entre si (71-75s). Não é uma característica real do
  algoritmo.
- [x] **Rodar nos datasets restantes do artigo** — parcialmente resolvido.
  Rodados com sucesso (20 runs, protocolo completo): **Data1, Data2, Data3,
  MPEG-7 BAS, Breast Tissue**. Somando aos já validados antes (Boat,
  Cone-Torus, Thyroid), são **8 de 12 datasets do artigo testados**.

  **Resultado (revisado após investigação mais profunda, ver item
  específico do Breast Tissue abaixo): 8/8 com Fuzzy-OPF ≥ OPF**, dentro
  do ruído estatístico esperado. Breast Tissue pareceu inicialmente uma
  exceção (~0.005 abaixo em 2 rodadas), mas uma terceira rodada com seeds
  totalmente independentes inverteu o sinal (+0.0015) -- confirmando que
  era ruído de amostra pequena, não um efeito real. Reproduz a
  propriedade central reivindicada pelo paper em todos os 8 datasets
  testados.
  Acurácias absolutas tendem a vir mais altas que o artigo em alguns casos
  (ex.: MPEG-7 BAS: nosso ~0.90 vs. artigo ~0.80), mas a direção
  qualitativa (Fuzzy-OPF raramente perde) se mantém na maioria.
  Resultados em `results/{data1,data2,data3,mpeg7_BAS,breast-tissue}/*.csv`.

  **Decisão do usuário: não rodar Four-Class e Landsat Satellite** (ficam
  fora do escopo definitivamente, não só "pendentes"). Breast Tissue teve
  uma discrepância menor: artigo diz 10 atributos, arquivo real tem 9.
  **Inatingíveis**: Electric Industrial Profiles e Electric Commercial
  Profiles são dados **privados** do artigo, sem fonte pública -- fora de
  alcance permanentemente. Ao todo, 8 de 12 datasets do artigo são
  considerados cobertos para os fins deste projeto.
- [x] **Investigar por que o Fuzzy-OPF fica consistentemente (~0.005, 2
  seeds) abaixo do OPF no Breast Tissue** — **resolvido, com correção
  importante depois de mais investigação**. A explicação original
  (validação de 20 amostras granular demais, curva de sigma completamente
  plana na validação) se confirmou -- mas persistiu mesmo testando k-fold
  CV (5 folds) e `k_max` maior (20 em vez de 5): a acurácia de validação
  continuou **idêntica bit-a-bit** entre as duas configurações, sinal de
  que, neste pool pequeno (~85 amostras), as previsões simplesmente não
  mudam com sigma/k_max nessa faixa -- não é algo que busca melhor ou
  validação melhor resolve.

  **A pista decisiva**: as duas rodadas de 20 runs originais (seed base 0 e
  1) se sobrepõem em 19 das 20 seeds internas (`seed = base_seed +
  run_id`), então não eram duas amostras independentes. Uma terceira
  rodada com seed base 100 (seeds 100-119, zero sobreposição) deu Fuzzy-OPF
  **acima** do OPF (+0.0015), invertendo o sinal das duas anteriores
  (-0.0056, -0.0046). **Conclusão final: é ruído estatístico de amostra
  pequena, não um efeito real e sistemático contra o Fuzzy-OPF** -- o
  sinal muda de direção conforme o conjunto de teste, sempre dentro de
  ±0.005, bem menor que o desvio padrão de cada rodada (~0.05). A "exceção"
  do Breast Tissue não é uma exceção real. `cv_folds` (k-fold) e o suporte
  a `k_max` alternativo em `sweep_sigma.py` ficam disponíveis e testados,
  mesmo não tendo sido a explicação final aqui -- podem servir em outros
  datasets pequenos no futuro. Resultados em
  `results/breast-tissue/{sigma_sweep_20260917T155218Z,sigma_sweep_20260918T042916Z,sigma_sweep_20260918T043513Z,20260918T044524Z}.csv`.
- [ ] **NOVO, motivado pelo achado acima**: para datasets pequenos (Boat,
  Breast Tissue, Data2, Data3, ...), considerar validação cruzada
  (k-fold) em vez do split único 60/20/20 para escolher hiperparâmetros --
  um conjunto de validação maior (efetivamente, via k-fold) teria mais
  chance de distinguir entre valores de sigma que hoje empatam por
  granularidade insuficiente.
- [x] **Endereçar o desbalanceamento de classes** no Thyroid — **resolvido
  com sucesso via SMOTE**, depois de uma tentativa negativa informativa.

  **Tentativa 1 (oversampling simples/duplicação)**: falhou. Balanceando
  totalmente as 3 classes (4004 amostras cada) por duplicação com
  reposição, o recall da classe minoritária "1" **piorou** (34.21% ->
  32.89%) e a acurácia geral caiu (0.7416 -> 0.7299). Explicação: o OPF
  compete por **topologia de grafo**, não por peso/gradiente -- duplicar
  um ponto o coloca exatamente na mesma posição do original, sem mudar a
  estrutura do grafo nem quem conquista as regiões de fronteira.

  **Tentativa 2 (SMOTE, pontos sintéticos interpolados)**: **sucesso
  claro**. Mesmo balanceamento total (4004 por classe), mas com pontos
  sintéticos gerados por interpolação entre uma amostra minoritária e um
  vizinho da mesma classe (`smote_oversample()`, k=5 vizinhos, sem
  dependência nova). Resultado: recall da classe "1" **quase dobrou**
  (34.21% -> 51.32%), classe "0" também melhorou (66.67% -> 72.73%),
  acurácia geral subiu **+5.16 pontos** (0.7416 -> 0.7931) com queda
  pequena e esperada na classe majoritária (97.45% -> 95.27%). Confirma a
  hipótese mecanística: pontos em posições NOVAS mudam a topologia do
  grafo perto da fronteira das classes raras, dando a elas território
  genuíno -- o que duplicatas exatas não conseguem fazer. **Contribuição
  publicável**: para classificadores da família OPF, SMOTE funciona onde
  oversampling simples falha (e até piora), por uma razão mecanística
  específica ao método (competição por topologia de grafo).
  `smote_oversample()`, `apply_balance()` (despachante configurável via
  YAML: `balance: {method: smote, k_neighbors: 5}`) e
  `experiments/compare_balance.py` (compara os 3 cenários) disponíveis e
  testados (15 testes, incluindo confirmação de que nenhum ponto
  sintético é duplicata exata de um original).
- [x] **Split estratificado**: `opfython.stream.splitter.split()` não
  estratifica por classe — pode importar em datasets desbalanceados como o
  Thyroid. **Resolvido**: `fuzzy_opf.datasets.stratified_split()`
  implementado e testado, disponível via `stratified: true` nas configs.

## Performance

- [x] **Paralelismo** — **parcialmente resolvido**. `run_hyperparam_search.py`
  agora roda GA, PSO e Random **em processos separados e concorrentes**
  (`ProcessPoolExecutor`, `parallel: true` por padrão) em vez de sequência
  -- as três buscas são independentes, então isso é paralelizável sem
  tocar no algoritmo de cada método. **Achado importante ao medir**: no
  Cone-Torus (rápido), o modo paralelo saiu **mais lento** que o
  sequencial (10,1s vs. 9,1s) -- cada processo paga um custo fixo de
  inicialização (reimportar numpy/opfython, aquecimento JIT do Numba) que
  não compensa quando a busca real leva só segundos. Só vale a pena em
  datasets caros (Thyroid), onde minutos de computação real ofuscam esse
  overhead -- documentado no código, `parallel: false` disponível pra
  desativar em datasets pequenos. **Ainda não feito**: paralelizar dentro
  de uma única busca (avaliação de agentes de UMA geração do GA/PSO em
  paralelo) e paralelizar o cálculo de distâncias par-a-par dentro do
  treino do Fuzzy-OPF em si -- o gargalo O(n²) sequencial mencionado pelo
  artigo continua lá dentro de cada método individual.
- [x] **Compartilhar cache de clustering entre métodos** em
  `run_hyperparam_search.py` — hoje GA, PSO e Random cada um recalcula o
  clustering uma vez (redundante quando `k_max` é fixo entre os três).
  **Resolvido**: `cluster_cache` compartilhado, testado (confirmado que
  um `k_max` calculado por um método é reaproveitado pelos outros dois).

## Novas funcionalidades / comparações

- [x] **Otimização Bayesiana via Optuna** — **implementado e testado**.
  `bayesian_search()` (TPE via Optuna, dependência nova adicionada ao
  `pyproject.toml`) segue a mesma interface de `genetic_search`/
  `pso_search`/`random_search`, incluindo suporte a `cv_folds` e
  `cluster_cache`. Diferente do GA/PSO, `n_evaluations` bate **exatamente**
  com `n_trials` (sem a inflação populacional que atrapalhou comparações
  anteriores) -- comparação de orçamento direta, sem distinção
  nominal/real. Integrado como 4º método em `run_hyperparam_search.py`
  (`ga`, `pso`, `random`, `bayesian`, todos rodando em paralelo). Testado
  no Cone-Torus: achou a mesma região boa de sigma que os outros três.

  **Teste no Thyroid com orçamento=40**: os 4 métodos convergiram pro
  mesmo resultado (val_acc=0.7559, sigma~1.11-1.16). O Bayesiano fez
  exatamente 40 avaliações (igual ao Random) e não demonstrou vantagem de
  eficiência amostral **com esse orçamento** -- consistente com o padrão
  já visto nessa investigação: quando o orçamento é grande o bastante pra
  achar o platô largo de sigma, qualquer método competente converge, e a
  comparação não discrimina.

  **Teste com orçamento pequeno (9 avaliações)**: resultado limpo e
  conclusivo. GA (18 avaliações reais) e PSO (12) ficaram presos no mesmo
  vale de ótimo local (sigma~0.92, val_acc=0.7376) já identificado antes,
  enquanto Random e Bayesiano (9 avaliações cada, batendo exatamente com
  o orçamento) acharam o platô bom (sigma~1.11-1.16, val_acc=0.7559) --
  **mesmo com MENOS avaliações que GA/PSO**. Confirma: em orçamentos
  pequenos, métodos populacionais (GA/PSO, com população forçosamente
  pequena pelo orçamento) são mais suscetíveis a ótimos locais do que
  métodos baseados em modelo (Bayesiano) ou até busca aleatória sem
  nenhuma inteligência -- mais avaliações não ajudam o GA/PSO aqui porque
  o problema é falta de diversidade populacional, não falta de tentativas.
  **Contribuição publicável**: para o Fuzzy-OPF no Thyroid, GA/PSO com
  orçamento pequeno arriscam convergência prematura; Bayesiano (ou até
  Random) são escolhas mais seguras nesse regime.
- [x] **Otimização multiobjetivo (NSGA-II)** — **implementado e testado**.
  `nsga2_search()` retorna o front de Pareto completo (não um único
  "melhor"), com dois objetivos: erro de validação e `k_max` (proxy
  determinístico de custo computacional -- tempo de relógio foi
  descartado como objetivo por ser ruidoso, 2-3x de variação run-a-run
  medida nesta própria investigação). Ambos objetivos avaliados a partir
  de um único treino por agente (memoizado por posição exata, evitando
  treinar 2x). Testado no Cone-Torus: o front colapsou inteiro em
  `k_max=2` -- resultado correto, não bug: nesse dataset fácil, `k_max`
  maior nunca dá acurácia melhor, então não existe trade-off real e o
  NSGA-II descobriu isso sozinho. 18 testes passando, incluindo
  verificação formal de que nenhum ponto do front é dominado por outro.

  **Teste no Thyroid (`k_max_bounds` largo, 1-100)**: front com apenas 3
  pontos distintos (k_max=1, 2, 8), cobrindo val_acc de 0.7494 a 0.7562
  (diferença de só 0.0068) -- confirma que **no Thyroid, k_max importa
  muito pouco pra acurácia**, diferente de sigma (que já sabíamos ser
  crítico, ver achado da multimodalidade). Recomendação prática direta:
  usar k_max pequeno (1-8) no Thyroid sem perda de acurácia, ganhando
  velocidade real (clustering escala com k_max). Resultado em
  `results/thyroid/pareto_20260918T201804Z.csv`.
- [ ] **CEM**: revisitar quando a `opytimizer` corrigir o bug de
  compatibilidade com NumPy 2.x (`cem_search` já existe, documentado como
  quebrado). **Reportado oficialmente**:
  https://github.com/recogna-lab/opytimizer/issues/10 (2026-09-21).
  Verificar periodicamente se foi corrigido antes de tentar de novo.
- [x] **Funções de pertinência alternativas** — **implementado, testado e
  investigado até conclusão definitiva**. `membership_kind` (linear,
  quadratic [Eq. 5 original], cubic, sigmoid) em `FuzzyOPF`, todas
  satisfazendo as mesmas condições de contorno (`F(rho_min)=sigma`,
  `F(rho_max)=1)` que a original -- só a forma da curva entre os dois
  pontos muda, garantindo comparação justa (confirmado por teste formal).

  **Teste 1 (hiperparâmetro fixo, sigma=0.6 para todas, Thyroid)**:
  quadratic=0.7416 melhor, sigmoid=0.7299 pior -- diferença de 0.0117.

  **Teste 2 (busca via GA própria pra cada forma, n_agents=15,
  n_iterations=5, 2 seeds, Thyroid)**: **as 4 formas convergiram pro mesmo
  resultado** -- todas acharam `sigma` na região 1.08-1.2 e deram
  acurácia de teste entre 0.7502 e 0.7551 (praticamente indistinguível).

  **Conclusão final**: a diferença do Teste 1 era um artefato de usar
  `sigma=0.6` fixo para todas -- bom para a quadrática, ruim para a
  sigmoide, não uma diferença real de capacidade entre as formas. **Com
  busca de hiperparâmetro adequada, a forma da curva de pertinência
  importa pouco** -- o que importa é achar o `sigma` certo. Achado
  publicável: reforça a importância de busca robusta de hiperparâmetro
  antes de comparar variações estruturais do método (mesmo tema do
  achado de multimodalidade do Thyroid). NOTA: por uma falha do script na
  época (corrigida -- CSVs agora incluem `membership_side`/
  `membership_kind` como colunas, e o nome do arquivo é sufixado com a
  forma), não foi possível identificar com certeza qual dos 4 CSVs
  originais corresponde a qual forma -- mas como o resultado é um empate
  entre as 4, isso não compromete a conclusão.
- [x] **Comparação sistemática de métricas de distância** (manhattan,
  chi_squared, bray_curtis, etc.) além da `log_squared_euclidean` padrão.
  **Resolvido**: `experiments/compare_distances.py` implementado e
  testado no Cone-Torus (mostrou diferença real: cosseno bem pior que as
  demais nesse dataset).
- [x] **Ensemble de Fuzzy-OPF** — **implementado e testado**.
  `EnsembleFuzzyOPF` (votação majoritária, com desempate determinístico)
  aceita qualquer lista de configurações de hiperparâmetros, ou pode ser
  montado diretamente a partir de um front de Pareto do `nsga2_search`
  (`EnsembleFuzzyOPF.from_pareto_front`) -- cada ponto do front já é, por
  definição, não-dominado, então é uma fonte de membros diversos e
  principiada, não arbitrária. Script `compare_ensemble.py` +
  config `ensemble_thyroid.yaml` prontos (compara ensemble vs. o melhor
  modelo único do mesmo front). 25 testes passando, incluindo verificação
  de que um ensemble de 1 membro reproduz exatamente as previsões desse
  modelo sozinho.

  **Testado no Thyroid** (front de 12 membros, `k_max_bounds=[1,100]`):
  single_best=0.7543 (k_max=8, sigma=1.125) vs. ensemble=0.7548 --
  diferença de +0.0005, essencialmente empate (bem dentro do ruído já
  observado, ~0.005-0.01, em toda essa investigação). Explicação: o front
  de Pareto do Thyroid tem variação pequena entre seus pontos (só 0.0068
  entre k_max=1 e k_max=8, achado do NSGA-II original), então os 12
  membros do ensemble são parecidos demais entre si para haver diversidade
  de erro que a votação possa corrigir -- ensemble só ganha força real
  quando os membros discordam de forma útil.

  **Testado também no MPEG-7 BAS** (70 classes, 180 atributos --
  estruturalmente bem diferente do Thyroid, front de 20 membros,
  `k_max_bounds=[1,150]`, `n_agents=20, n_iterations=15`): single_best=0.9165
  (k_max=1, sigma=0.813) vs. ensemble=0.9165 -- **empate exato**.

  **Conclusão final (2 datasets estruturalmente diferentes, mesmo
  padrão)**: o ensemble construído a partir do front de Pareto do
  Fuzzy-OPF **não supera o melhor modelo único** -- não é peculiaridade do
  Thyroid, é um padrão observado (ainda não testado nos outros 6
  datasets). Hipótese consistente nos dois casos: os pontos de um front de
  Pareto são, por construção, próximos em desempenho (senão não seriam
  Pareto-ótimos todos ao mesmo tempo), então tendem a errar nos mesmos
  casos -- a fonte de diversidade que faria um ensemble valer a pena
  (membros que discordam de forma útil) não vem naturalmente de um front
  de Pareto. Achado honesto e informativo para o artigo: motiva a técnica,
  testa rigorosamente, e reporta um resultado negativo bem explicado --
  mesmo padrão de rigor já usado no achado do oversampling simples vs.
  SMOTE.
## Infraestrutura / publicação

- [ ] **Publicar no PyPI** — hoje só instala via clone + `pip install -e .`.
- [ ] **Propor PR para a própria `opfython`** — o Fuzzy-OPF preencheria uma
  lacuna real na lib (que só tem Supervised/Unsupervised/KNN/Semi-Supervised
  OPF).

## Segunda rodada de sugestões externas (2026-09-22) -- tendências TFS + análise cruzada

Registrado a partir de duas listas adicionais (uma revisão de esforço por
item, e uma análise de tendências da IEEE Transactions on Fuzzy Systems +
literatura recente do grupo do Prof. Papa). Boa parte já estava no roteiro
acima (Fuzzy OPF-AD, Active Learning, seleção de protótipos, DE/GWO [já
implementado], pertinência adaptativa, on-the-fly, GPU/Spark, incremental)
-- só o que é genuinamente novo é listado aqui, ordenado por esforço.

- [ ] **9. Robustez a ruído de rótulos** -- corromper X% dos rótulos de
  treino (5% a 30%) e medir quantitativamente se a pertinência fuzzy atua
  como "amortecedor" contra protótipos ruidosos conquistando grandes
  regiões do grafo, comparando Fuzzy-OPF vs. OPF padrão sob ruído
  crescente. Esforço baixo -- reaproveita toda a infraestrutura de
  experimento já pronta (só precisa de uma função de corrupção de rótulos
  antes do treino).
- [ ] **10. API estilo scikit-learn (`predict_proba`)** -- expor
  `predict_class_scores()` (já implementado para AUC-ROC) sob a convenção
  `predict_proba`, e formalizar `fit`/`predict`/`predict_proba` como
  interface pública documentada. Esforço baixo -- quase todo o código já
  existe, é principalmente documentação/polimento de API. Complementa o
  item de publicar no PyPI já registrado.
- [ ] **11. Explicabilidade via caminho ótimo (XAI path-based)** -- o
  caminho de conquista (predecessores) já é calculado internamente
  durante o treino; falta extrair e visualizar: para uma predição, mostrar
  o protótipo de origem, o caminho percorrido no grafo, e o peso fuzzy
  acumulado ao longo da trajetória, como justificativa da decisão.
  Esforço médio.
- [ ] **12. Pertinência por ambiguidade de fronteira (entropia de
  vizinhança de rótulos)** -- refinamento mais concreto e testável do
  item 5 do roteiro anterior (pertinência adaptativa): em vez de entropia
  local genérica, usar especificamente a MISTURA DE RÓTULOS na
  vizinhança de cada nó -- se um nó está numa região densa mas cercada por
  vizinhos de classes opostas (fronteira de decisão), sua pertinência cai,
  independente da densidade pura. Tema de alta relevância atual na IEEE
  TFS (pertinência sob incerteza de fronteira / possibilistic clustering).
  Esforço médio -- mais bem definido que o item 5 genérico, bom candidato
  a vir antes dele.
- [ ] **13. Distâncias não-Euclidianas (Mahalanobis, geodésica,
  Wasserstein)** -- Mahalanobis pode já estar registrada na `opfython`
  (já testamos várias métricas dela em `compare_distances.py`, conferir se
  está na lista); geodésica e Wasserstein exigiriam implementação do
  zero. Esforço parcial-baixo para Mahalanobis, alto para as outras duas.

**Fora de escopo para este trabalho (registradas apenas como trabalho
futuro no texto do artigo, não para implementar agora)**: Fuzzy OPF sobre
features profundas/deep learning (pipeline totalmente diferente, exige
rede neural e dataset de imagem que não temos), Fuzzy OPF Federado
(sem cenário multi-nó real para testar), formulação quantum-inspired/PUBO-
QUBO para seleção de protótipos (foge completamente do escopo, é outro
projeto de pesquisa inteiro).

**Estrutura de publicação sugerida** (concordância com a análise externa):
Plano A (fechar o artigo atual) = Fuzzy-OPF + prova teórica de suavidade +
validação por meta-heurísticas (GA/PSO/DE/GWO/Bayesiano/NSGA-II) + KD-tree
-- já está pronto, sem pendência de código. Plano B (artigo de extensão
futuro, separado) = Fuzzy OPF-AD (detecção de anomalias) OU Fuzzy Active
Learning -- não tentar espremer no artigo atual.

## Roteiro para maior impacto internacional (2026-09-22, análise de sugestões externas)

Registrado a partir de uma lista de diretrizes de alto valor para
publicações futuras. Ordenado por esforço estimado (mais fácil primeiro),
com status do que já foi feito vs. o que é escopo novo.

- [x] **Prova formal de preservação/quebra das garantias teóricas** --
  já feito, ver `docs/smoothness_proof.tex` (item 6 da lista de revisão
  externa anterior).
- [x] **Meta-heurísticas para hiperparâmetros (parcial)** -- GA, PSO,
  Bayesiano (Optuna) e NSGA-II já implementados e testados. Faltam
  Differential Evolution e Grey Wolf Optimizer especificamente
  (a `opytimizer` provavelmente já os tem prontos -- baixo esforço).
- [x] **1. Adicionar Differential Evolution e Grey Wolf Optimizer** --
  **implementado e testado**. `de_search()` e `gwo_search()` seguem
  exatamente a mesma interface de `genetic_search`/`pso_search`
  (reaproveitam `_run_metaheuristic_search` sem duplicar lógica).
  Integrados como métodos 5 e 6 em `run_hyperparam_search.py` (agora
  compara GA, PSO, Random, Bayesiano, DE, GWO -- 6 métodos). Testado no
  Cone-Torus: os 6 convergem pra mesma região boa de sigma. Ajuste
  importante: `max_workers` agora limitado ao número de núcleos da
  máquina (antes seria `len(method_names)`, que com 6 métodos
  sobrecarregaria uma máquina de 4 núcleos) -- métodos extras entram na
  fila automaticamente, não são descartados. 35 testes passando. Ainda
  não rodado no Thyroid para ver se DE/GWO reproduzem o mesmo padrão de
  "presos em ótimo local com orçamento pequeno" que GA/PSO mostraram.
- [ ] **2. Fuzzy OPF para Detecção de Anomalias (Fuzzy OPF-AD)** -- bom
  encaixe com o que já existe: nós com pertinência muito baixa e custo de
  caminho desproporcional já são, implicitamente, os "outliers" que o
  Fuzzy-OPF identifica -- falta formalizar como tarefa de detecção
  (threshold sobre pertinência/custo, métricas de detecção de anomalia
  em vez de classificação). Esforço médio.
- [ ] **3. Fuzzy OPF para Active Learning / Semi-supervisionado** -- usar
  o grau de pertinência fuzzy já calculado como critério de incerteza
  para seleção de amostras a rotular (amostras de menor pertinência =
  maior ambiguidade = mais informativas para rotular). Esforço médio,
  reaproveita a pertinência que já calculamos.
- [ ] **4. Seleção de protótipos via meta-heurística** -- diferente de
  ajustar hiperparâmetros (k_max, sigma): usar GA/PSO para escolher quais
  amostras específicas viram protótipos, não só onde cortar o MST.
  Formulação nova, esforço médio-alto.
- [ ] **5. Funções de pertinência adaptativas** (entropia local da
  vizinhança, perturbação de vizinhança, ou Type-2 Fuzzy Sets) -- diferente
  de `membership_kind`/`membership_source` (que já testamos, ambos
  estáticos por amostra): pertinência que muda dinamicamente com a
  incerteza local. Type-2 Fuzzy especificamente é mudança de framework
  matemático, não ajuste incremental. Esforço alto.
- [ ] **6. Fuzzy OPF sem pré-clustering (custo fuzzy on-the-fly)** --
  eliminar a fase separada de estimativa de densidade, incorporando a
  incerteza diretamente na competição em tempo de execução. O KD-tree
  (item 7 da lista anterior) acelera o pré-clustering mas não o elimina --
  isso é mudança arquitetural real. Esforço alto.
- [ ] **7. Fuzzy OPF incremental para data streams** -- atualizar o grafo
  e a matriz de pertinência em tempo real sem retreinar do zero. Mais
  distante do que já existe (tudo assume batch); exigiria repensar a
  estrutura de dados do grafo. Esforço alto.
- [ ] **8. Fuzzy OPF distribuído/GPU (CUDA/Spark)** -- validar
  escalabilidade para milhões de instâncias. Dificultado pela natureza
  sequencial/gulosa do algoritmo de competição (cada nó conquistado
  depende do estado atual do heap, resistente a vetorização GPU-style);
  também exigiria datasets que não temos. Esforço muito alto -- candidato
  a ficar fora do escopo desse trabalho, registrado como trabalho futuro
  no próprio artigo em vez de implementado.

## SMOTE + busca robusta nos demais datasets desbalanceados -- 2026-09-22

- [x] **Testado SMOTE + busca via GA robusta (n_agents=15) em Cone-Torus,
  Data3 e Breast Tissue** -- motivado pela pergunta "será que o ganho do
  SMOTE no Thyroid generaliza pros outros datasets desbalanceados?".
  Primeiro caracterizamos o desbalanceamento real de cada um dos 8
  datasets (razão maioria/minoria): Boat (~1.03x), Data1 (~1.25x), Data2
  (~1.16x) e MPEG-7 BAS (1.00x, perfeitamente balanceado) são
  essencialmente balanceados -- SMOTE não se aplica. Cone-Torus (~2.3x),
  Breast Tissue (~1.6x) e Data3 (~3.8x) têm desbalanceamento leve a
  moderado. Thyroid (~30x) é o único com desbalanceamento severo.

  **Resultado**:
  - **Breast Tissue**: sem mudança perceptível (0.72 com SMOTE, dentro da
    faixa 0.70-0.73 já observada sem SMOTE em múltiplas seeds).
  - **Cone-Torus**: OPF e Fuzzy-OPF melhoraram quase igual (+0.0112 e
    +0.0109) -- mas a comparação não é limpa (orçamento de busca também
    mudou, n_iterations=10 vs. 30 da config original), e o ganho não foi
    específico do Fuzzy-OPF.
  - **Data3**: SMOTE **não ajudou** -- Fuzzy-OPF caiu de 0.9935 para
    0.9916 (empatando com OPF, que subiu ligeiramente); a pequena
    vantagem que existia sem SMOTE desapareceu.

  **Conclusão**: o benefício do SMOTE **não generaliza** para qualquer
  dataset desbalanceado -- é proporcional à combinação de (1) severidade
  real do desbalanceamento e (2) espaço de melhora disponível (acurácia
  longe do teto). No Data3, a acurácia já estava perto de 99% sem SMOTE --
  não havia "problema" real para o SMOTE resolver, e a técnica introduziu
  ruído em vez de ajudar. No Thyroid, havia desbalanceamento severo E
  acurácia baixa o suficiente para ter espaço de melhora real, condição
  que nenhum dos outros 7 datasets reproduz. **SMOTE não é correção
  universal para desbalanceamento no Fuzzy-OPF** -- funciona
  especificamente no regime "desbalanceamento severo + acurácia baixa",
  não em qualquer grau de desbalanceamento. Resultados em
  `results/{cone-torus,data3,breast-tissue}/2026*_quadratic.csv`.

## Verificação contra a implementação de referência (LibOPF, C) -- 2026-09-22

- [x] **Compilado o LibOPF (C, https://github.com/jppbsi/LibOPF) e rodado
  diretamente no Thyroid**, mesma seed=0, mesmo split 60/20/20, mesma
  normalização, pra verificar se o gap de ~23 pontos contra o artigo
  original (97.14%) vinha de uma diferença entre `opfython` (Python) e
  LibOPF (C) que a investigação da métrica não tinha descartado.

  **Achado no processo**: o formato de texto nativo do LibOPF usa rótulos
  **1-indexados** (1,2,3...), diferente do nosso `thyroid.txt` (0-indexado,
  0,1,2). Sem corrigir isso, o `opf_Accuracy` do C (que itera
  `for i=1; i<=nlabels`) ignoraria silenciosamente toda a classe rotulada
  0 -- que por acaso é a majoritária (92% dos dados). Corrigido no script
  de exportação (`export_to_libopf.py`, desloca rótulos +1). Confirmado
  também que a fórmula do `opf_Accuracy` em C é **idêntica** à do
  `opfython` (mesmo balanceamento por classe) -- lendo o código-fonte C
  diretamente, não só testando.

  **Resultado**: LibOPF (C) = 73.70% de acurácia balanceada no Thyroid --
  **praticamente idêntico** ao nosso OPF/Fuzzy-OPF via `opfython`
  (~74-75%), inclusive com a matriz de confusão batendo quase exatamente
  (recall da classe minoritária: 34.21%, o mesmo número que aparece
  repetidamente em toda essa investigação).

  **Conclusão definitiva**: o port em `opfython` é **fiel** à
  implementação C de referência -- não existe bug de tradução nem
  diferença de comportamento entre as duas no Thyroid. O gap remanescente
  contra o número reportado no artigo original (~97%) **não é explicado
  pela nossa implementação** -- é atribuível a alguma diferença de
  dataset/pré-processamento do artigo que não foi possível reconstruir
  (já sabíamos que eles usam 2 classes, não 3; mesclar não foi suficiente
  para fechar todo o gap sozinho, ver investigação anterior). Encerra essa
  linha de investigação com uma conclusão forte: nossa implementação está
  correta e verificada contra a referência oficial.



- [x] **Levantamento inicial feito**. Achados principais:
  - **"Handling Imbalanced Datasets Through Optimum-Path Forest"** (Passos,
    Jodas, Ribeiro, de Souza, Papa -- mesmo grupo do Fuzzy-OPF original)
    já propôs técnicas de balanceamento **nativas do OPF**: O²PF
    (oversampling via clustering + distribuição Gaussiana por cluster) e
    OPF-US (undersampling via pontuação de importância no processo de
    competição). Resultado deles: **OPF-US (undersampling nativo)
    geralmente supera SMOTE genérico** no OPF padrão. Isso não invalida
    nosso achado (SMOTE > duplicação simples no Fuzzy-OPF, com explicação
    mecanística de topologia de grafo), mas expõe uma comparação que
    falta: **testar OPF-US contra SMOTE no nosso Fuzzy-OPF**, já que o
    grupo original sugere que undersampling nativo pode ser ainda melhor.
  - Encontrada aplicação prática do Fuzzy-OPF (IoT security monitoring,
    hit rate 98-99%) e um precedente do próprio grupo (Probabilistic OPF)
    usando metaheurísticas (Bat, Firefly, PSO, Nelder-Mead) para tuning de
    hiperparâmetros -- confirma que a prática é estabelecida no grupo,
    mas nenhum trabalho encontrado faz a comparação GA/PSO/Bayesiano/
    Random com contagem real de avaliações, nem caracteriza
    multimodalidade do espaço de busca, nem usa NSGA-II para o trade-off
    k_max-vs-acurácia no contexto do (Fuzzy-)OPF -- essas parecem
    contribuições originais deste projeto.
- [x] **Implementar OPF-US e comparar contra SMOTE no Thyroid** —
  **resolvido**. `opf_us_undersample()` (generalização multiclasse da
  técnica de Passos et al. 2022) testada lado a lado com SMOTE.
  Resultado: **os dois melhoram sobre o baseline por margem parecida**
  (SMOTE +0.0516, OPF-US +0.0473 -- dentro do que seria empate sem teste
  estatístico formal), mas com **padrões de erro opostos**. SMOTE
  (34%->51% recall na minoritária, 97%->95% na majoritária) preserva mais
  a classe majoritária; OPF-US (34%->68% na minoritária, mas 97%->73% na
  majoritária) prioriza recall equilibrado à custa de descartar 97% da
  classe majoritária (4004->103 amostras). **Não há vencedor absoluto** --
  depende se o caso de uso prioriza recall equilibrado (OPF-US) ou
  preservar a classe majoritária com menor perda de dados (SMOTE). Essa
  caracterização do trade-off é a resposta direta à pergunta de revisor
  identificada na pesquisa de trabalhos relacionados.
- [ ] Levantamento mais aprofundado (busca sistemática, não só
  exploratória) antes de submeter -- confirmar que não há outro trabalho
  cobrindo exatamente a combinação (Fuzzy-OPF + comparação de
  metaheurísticas + NSGA-II) antes de reivindicar ineditismo no artigo.

## Sugestões de revisão externa (2026-09-20, análise de lista de melhorias)

Registrado a partir de uma lista de sugestões de melhoria pro artigo,
organizadas do mais simples pro mais complicado (ordem de implementação
sugerida). Ver conversa de 2026-09-20 para a análise completa de
"já feito / acionável / fora de alcance" por item.

- [x] **1. Teste de Wilcoxon pareado** — **implementado e concluído em
  todos os 8 datasets validados**. `experiments/wilcoxon_test.py`
  (`scipy.stats.wilcoxon`, pareado por run/seed + tamanho de efeito
  `r=|z|/sqrt(N)`), testado, 26 testes passando.

  **Resultado consolidado** (só significativo a `p<0.05` em 2 de 8):

  | Dataset | Ganho médio | p-valor | Efeito |
  |---|---|---|---|
  | Boat | +0.0020 | 0.3173 | pequeno |
  | Cone-Torus | +0.0000 a +0.0021 | 0.17-0.89 | negligível-médio |
  | **Data1** | +0.0010 | **0.0422** | médio (significativo) |
  | Data2 | +0.0027 | 0.2733 | pequeno |
  | Data3 | +0.0031 | 0.1088 | médio |
  | **MPEG-7 BAS** | +0.0015 | **0.0007** | grande (significativo) |
  | Breast Tissue | -0.0056 a +0.0015 | todos n.s. | pequeno-médio (ruído, ver
  achado do overlap de seeds) |
  | Thyroid (20 runs, sigma=1.15 fixo -- ver nota abaixo) | -0.0006 | 0.5503
  | pequeno (empate estatístico) |

  **Nota importante**: nenhuma das buscas anteriores no Thyroid tinha
  rodado o protocolo oficial de 20 runs (sempre `n_runs=2`, por custo) --
  os resultados de Thyroid usados em todo o resto do backlog vêm de
  comparações com N pequeno. Rodamos uma vez com hiperparâmetro FIXO
  (`k_max=20, sigma=1.15`, sem busca via GA -- ~57min para 20 runs, viável
  onde 20 runs COM busca seriam horas) especificamente para ter uma
  amostra válida para o Wilcoxon. Resultado:
  `results/thyroid/20260921T160914Z_quadratic.csv`.

  **Conclusão para o artigo**: a afirmação defensável não é "Fuzzy-OPF é
  melhor que OPF" de forma geral (os dados não sustentam isso com rigor
  estatístico) -- é "Fuzzy-OPF nunca é significativamente pior, às vezes é
  significativamente melhor (efeito grande no MPEG-7 BAS), e na maioria
  dos casos os dois são estatisticamente indistinguíveis". Mais sutil, mas
  mais defensável e resistente a crítica de revisor do que uma alegação
  inflada de superioridade geral.
- [x] **2. AUC-ROC** -- **implementado e testado**. `FuzzyOPF.
  predict_class_scores()` faz uma varredura completa (sem poda) pra obter
  um score de confiança por classe (não só a vencedora) -- confirmado por
  teste formal que a classe de maior score sempre bate com `predict()`.
  `auc_roc_test.py` compara Fuzzy-OPF contra um "OPF equivalente"
  (`sigma=1`, matematicamente igual ao OPF puro pela própria teoria do
  artigo) via AUC-ROC macro (one-vs-rest) + recall por classe. 27 testes
  passando.

  **Achado interessante no Cone-Torus**: mesmo quando a decisão final
  (accuracy/recall) é idêntica entre sigma=0.9 e sigma=1 (platô plano já
  visto em todo teste nesse dataset), a **AUC-ROC difere** (0.8945 vs.
  0.9047) -- a pertinência muda a confiança relativa mesmo sem mudar qual
  classe "vence", algo que accuracy/recall (que só olham a decisão final)
  não capturam. AUC-ROC revela uma diferença que outras métricas escondem.

  **Testado no Thyroid** (sigma=1.15, sem balanceamento): AUC-ROC
  OPF=0.8598 vs. Fuzzy-OPF=0.8544 (-0.0054, consistente com o empate
  estatístico já visto no Wilcoxon nesse mesmo sigma). Por classe: recall
  da classe 0 melhora bastante com Fuzzy-OPF (0.6364->0.7273, +9pp), classe
  2 piora um pouco (0.9850->0.9775, -0.75pp), classe 1 (minoritária, mais
  difícil) empata exatamente (0.3421 nos dois). **Conclusão**: não é uma
  derrota uniforme, é um trade-off de recall entre classes -- reforça a
  leitura madura já estabelecida pelo Wilcoxon (sem balanceamento, os dois
  são essencialmente equivalentes no Thyroid, com nuances pequenas e
  específicas por classe, não uma vantagem clara de um lado).
- [x] **3. Baselines externos** (SVM RBF, Random Forest, XGBoost, k-NN) --
  **implementado, com um episódio de investigação importante no meio**.
  `compare_baselines.py` (GridSearchCV pra cada baseline, comparação justa
  contra padrões não-otimizados) + configs pros 8 datasets validados.

  **Episódio de investigação (Thyroid)**: a primeira rodada mostrou um gap
  absurdo (~20+ pontos) entre Fuzzy-OPF (~0.75) e os baselines externos
  (94-99%), grande demais pra ser real. Investigação sistemática, na
  ordem testada: (1) métrica de distância (`log_squared_euclidean` vs.
  `euclidean`) -- descartada, mesma acurácia; (2) normalização -- descartada,
  sem normalizar ficou PIOR (0.6768), não melhor; (3) conversão de dados --
  descartada, valores das colunas batem exatamente com o formato padrão
  do `ann-thyroid`; (4) `opfython` puro (sem nada nosso) -- ainda baixo
  (0.6776), afastando a hipótese de bug no nosso port; (5) protótipos por
  classe -- desproporcionalmente altos nas minoritárias (44.7% e 88.7%
  das amostras viraram protótipo), sugerindo sobreposição entre classes,
  mas descartado como causa principal; (6) duplicatas de features com
  rótulos conflitantes -- zero encontradas, descartada; (7) 2 vs. 3
  classes (o artigo usa 2, nós usamos 3) -- mesclando pra 2 classes,
  melhora modesta (0.7865), não decisiva; (8) diferença estrutural
  `opfython` (Python) vs. `LibOPF` (C, usado no artigo original) via
  empates de distância ("tie-zones", mencionado no próprio artigo na
  seção de Discussão) -- descartada, só 0.3% dos pares tinham distância
  empatada no Thyroid.

  **Causa raiz real (achada na tentativa 9)**: `opf_accuracy()` da
  `opfython` não é acurácia simples -- é uma **métrica balanceada por
  classe** (soma taxas de falso positivo/negativo por classe, normalizada
  pelo tamanho de cada classe, depois tira a média -- convenção
  estabelecida na literatura de OPF, não um bug). Toda a investigação
  desse projeto usou essa métrica para Fuzzy-OPF/OPF, mas
  `compare_baselines.py` originalmente usava **acurácia simples**
  `(preds==y_test).mean()` para os baselines sklearn -- uma comparação de
  métricas diferentes, não um problema real de desempenho. Confirmado
  isolando: mesmo modelo, mesmos dados, `opf_accuracy`=0.6776 vs. acurácia
  simples=0.9208. **Corrigido**: `compare_baselines.py` agora usa
  `opf_accuracy()` uniformemente para todos os métodos.

  **Resultado final, correto, no Thyroid (20 runs)**:

  | Método | Acurácia balanceada média |
  |---|---|
  | Random Forest | 0.9883 |
  | XGBoost | 0.9877 |
  | SVM (RBF) | 0.8921 |
  | **Fuzzy-OPF** | **0.7477** |
  | k-NN | 0.7101 |

  **Conclusão honesta para o artigo**: métodos baseados em árvore (RF,
  XGBoost) dominam datasets com desbalanceamento severo como o Thyroid --
  padrão bem documentado na literatura de ML geral, não uma fraqueza
  específica do Fuzzy-OPF. Fuzzy-OPF **supera o k-NN** (o baseline
  conceitualmente mais próximo, também baseado em distância/instância) --
  resultado legítimo dentro da categoria. O gap pro RF/XGBoost é real, mas
  já sabemos reduzi-lo: combinar com SMOTE (que já levou Fuzzy-OPF de
  0.7416 para 0.7931 no Thyroid, ver achado de desbalanceamento) é o
  próximo passo natural antes de reportar essa comparação no artigo.

  **Pendente**: rodar `compare_baselines.py` (com a métrica corrigida) nos
  outros 7 datasets (Boat, Cone-Torus, Data1/2/3, Breast Tissue, MPEG-7
  BAS) -- o resultado inicial (métrica errada) foi descartado para todos,
  não só o Thyroid.
- [x] **4. Teste de Friedman + post-hoc Nemenyi** -- **implementado e
  concluído**. `friedman_nemenyi_test.py` (`scipy.stats.friedmanchisquare`
  + `scikit_posthocs.posthoc_nemenyi_friedman`) comparando os 5 métodos
  (Fuzzy-OPF, SVM-RBF, Random Forest, k-NN, XGBoost) nos 8 datasets
  validados, usando a acurácia média de cada um (mesma correção de métrica
  do item 3, `opf_accuracy` uniforme).

  **Matriz de acurácia média (8 datasets x 5 métodos)**:

  | Dataset | Fuzzy-OPF | k-NN | Random Forest | SVM-RBF | XGBoost |
  |---|---|---|---|---|---|
  | Boat | 0.9894 | 0.9660 | 0.9672 | 0.9923 | 0.9244 |
  | Breast Tissue | 0.6980 | 0.6992 | 0.7925 | 0.7467 | 0.7944 |
  | Cone-Torus | 0.8520 | 0.8677 | 0.8574 | 0.8714 | 0.8384 |
  | Data1 | 0.9933 | 0.9931 | 0.9907 | 0.9934 | 0.9890 |
  | Data2 | 0.9590 | 0.9827 | 0.9816 | 0.9764 | 0.9793 |
  | Data3 | 0.9924 | 0.9904 | 0.9945 | 0.9944 | 0.9836 |
  | MPEG-7 BAS | 0.9043 | 0.8844 | 0.9046 | 0.9111 | 0.8567 |
  | Thyroid | 0.7477 | 0.7101 | 0.9883 | 0.8921 | 0.9877 |

  **Ranks médios** (1=melhor): SVM-RBF 2.00, Random Forest 2.25, k-NN
  3.38, **Fuzzy-OPF 3.50**, XGBoost 3.88.

  **Friedman: statistic=8.70, p=0.0691 -- NÃO significativo a α=0.05**
  (por pouco). Post-hoc Nemenyi não se aplica (Friedman não rejeitou a
  hipótese nula de igualdade entre os métodos).

  **Conclusão honesta para o artigo**: com 8 datasets, não há evidência
  estatística suficiente pra afirmar que os 5 métodos diferem de forma
  geral -- resultado limítrofe (p=0.069, perto de 0.05), não uma vitória
  clara de ninguém. Isso é consistente com o padrão observado
  dataset-a-dataset: Fuzzy-OPF vence/empata em datasets balanceados
  (Boat, Data1, MPEG-7 BAS) mas perde feio em desbalanceados/pequenos
  (Thyroid, Breast Tissue) -- a média dos ranks "esconde" essa
  variabilidade real entre datasets. Nota metodológica para o artigo: N=8
  datasets é pequeno para o Friedman ter bom poder estatístico (a
  literatura geralmente recomenda 10+); rodar Four-Class e/ou Landsat
  (fora do escopo por decisão do usuário, ver histórico) teria dado mais
  poder, mas não foi perseguido.

  **Atualização (2026-09-22)**: migrado de `scipy`/`scikit-posthocs`
  (implementação manual) para `Statys`
  (https://github.com/gugarosa/statys), biblioteca do mesmo
  autor/grupo por trás da `opfython`/`opytimizer` já usadas no projeto --
  mantém consistência metodológica com o resto do toolchain. Resultado
  numérico idêntico (chi-quadrado=8.70), mais o F de Iman-Davenport
  (2.61, frequentemente preferido sobre o qui-quadrado nesse tipo de
  comparação) e, principalmente, um **diagrama de diferença crítica**
  real (`plot_critical_difference`, o gráfico padrão da literatura
  Demšar 2006) que a versão manual não gerava -- confirma visualmente
  que os 5 métodos ficam conectados por uma única barra (nenhum par
  difere significativamente), consistente com o Friedman não
  significativo.
- [x] **5. Pertinência via Fuzzy C-Means (FCM)** -- **implementado e
  testado**. `membership_source` (`"density"`, padrão, Eq. 3 original;
  `"fcm"`, clustering FCM nas features cruas via `scikit-fuzzy`) em
  `FuzzyOPF` -- diferente do `membership_kind` (que só muda a curva de
  mapeamento): isso muda o que está sendo medido, não só como é mapeado
  para [sigma, 1]. Pertinência FCM = grau de pertinência máximo de cada
  amostra ao seu próprio cluster (dentre `fcm_n_clusters`, padrão = número
  de classes), normalizado e passado pela mesma curva de
  `membership_kind` (reuso de código via `_apply_membership_curve`, sem
  duplicar a lógica das 4 formas). Ambas as fontes satisfazem as mesmas
  condições de contorno (teste formal). Script
  `compare_membership_sources.py` + configs para os 8 datasets prontos.
  30 testes passando.

  **Testado nos 8 datasets validados** (mesmo sigma fixo por dataset já
  usado nos testes de baseline):

  | Dataset | Density | FCM | Diferença |
  |---|---|---|---|
  | Boat | 0.9688 | 0.9688 | 0.0000 |
  | Cone-Torus | 0.8668 | 0.8668 | 0.0000 |
  | Data1 | 0.9969 | 0.9969 | 0.0000 |
  | Data2 | 0.9444 | 0.9566 | +0.0122 |
  | Data3 | 0.9732 | 0.9732 | 0.0000 |
  | Breast Tissue | 0.7677 | 0.7677 | 0.0000 |
  | MPEG-7 BAS | 0.9165 | 0.9165 | 0.0000 |
  | Thyroid | 0.7543 | 0.7575 | +0.0032 |

  **Conclusão**: empate exato em 6 de 8 datasets -- confirma, de um ângulo
  diferente, o mesmo padrão já visto com `membership_kind`: uma vez que
  `sigma` está bem ajustado, a fonte de pertinência (densidade vs. FCM)
  quase não importa. A hipótese de que o FCM ajudaria especificamente no
  Thyroid (por captar melhor a estrutura de classes que a densidade,
  distorcida pelo desbalanceamento) **não se confirmou** -- diferença de
  só +0.0032, dentro do ruído. Única exceção com diferença real é o Data2
  (+0.0122), pequena demais para mudar a conclusão geral. Reforça:
  **sigma domina o comportamento do Fuzzy-OPF; as escolhas estruturais de
  como calcular/mapear pertinência são, na prática, secundárias** -- um
  achado consistente que já apareceu de duas formas independentes
  (`membership_kind` e `membership_source`) nessa investigação.

  **Gaussiana real**: não implementada (escopo do item ficou em FCM, que
  já é uma mudança estrutural suficiente para o objetivo de "pertinência
  calculada de forma diferente, não só mapeada diferente"); pode ser um
  item futuro separado se quiser mais uma fonte de comparação.
- [x] **6. Formalizar teoricamente a quebra da propriedade "smooth" da
  função de custo** -- **concluído**. Prova formal em
  `docs/smoothness_proof.tex`, ancorada exatamente na implementação real
  (`current_cost = membership * max(heap.cost[p], weight)`, convenção
  "target"). Resultados provados:
  - **Teorema principal**: para `sigma < 1`, a função de custo fuzzy NÃO
    é smooth em geral -- identifica a condição exata de quebra:
    `w(t,u) <= f'(caminho)` E `F_Theta(u) < 1` simultaneamente (comum na
    prática, justamente para as amostras mais "típicas"/de alta
    densidade).
  - **Corolário**: em `sigma = 1`, a suavidade é recuperada exatamente --
    justificativa formal (não só numérica) de por que o artigo afirma que
    Fuzzy-OPF degenera pro OPF padrão nesse ponto.
  - **Consequência algorítmica**: a guarda `BLACK` (que corrigimos no bug
    do ciclo infinito) é *necessária* no Fuzzy-OPF, nunca redundante como
    no OPF padrão -- conecta a prova formal diretamente ao bug real que
    encontramos e corrigimos.
  - **Ressalva importante de honestidade científica**: com a guarda
    restaurada, o algoritmo termina e produz uma floresta válida (sem
    ciclos), mas isso NÃO restaura a garantia clássica de otimalidade
    global ao estilo Dijkstra -- vira uma heurística gulosa bem definida
    sobre o grafo ponderado por fuzzy, não um algoritmo com a mesma
    garantia teórica do OPF padrão. Distinção importante para não
    superestimar o que foi provado.
  - **Discussão conectando com achado empírico anterior**: a perda de
    suavidade oferece uma explicação teórica plausível (não provada como
    causa única) para a multimodalidade do espaço de busca de `sigma` que
    encontramos empiricamente no Thyroid -- uma boa unificação entre teoria
    e experimento pro artigo.
  - **Nota sobre `membership_side`**: a mesma condição de quebra vale para
    a convenção "source" (troca só qual nó entra na fórmula) -- as duas
    convenções são igualmente não-suaves; a diferença empírica entre elas
    está em qual caminho o processo não-suave acaba preferindo, não em se
    a suavidade se mantém.
- [x] **7. Subamostragem/aproximação de densidade (KD-trees)** --
  **implementado e testado**. `membership_source="density_kdtree"` em
  `FuzzyOPF`: reproduz a fórmula exata de densidade da `opfython`
  (kernel Gaussiano/Parzen, `opfython.subgraphs.knn.KNNSubgraph.
  calculate_pdf`, lida diretamente do código-fonte pra garantir
  fidelidade), mas busca os k-vizinhos-mais-próximos via
  `scipy.spatial.cKDTree` (O(n log n)) em vez da busca por força bruta
  O(n²) da `opfython`.

  **Importante**: isso é um speedup **exato**, não uma aproximação --
  `log_squared_euclidean` (métrica padrão do projeto) é uma transformação
  monotônica da distância Euclidiana, então os vizinhos encontrados via
  KD-tree (Euclidiana) são garantidamente os mesmos que a busca por força
  bruta encontraria com a métrica configurada; só a distância final (mais
  barata, O(k) por nó) é calculada com a métrica real. Confirmado por
  teste formal: diferença **zero** entre os valores de densidade dos dois
  caminhos.

  **Benchmark** (dataset sintético do tamanho do treino do Thyroid,
  n=4320, d=21, k_max=20): 57,14s (força bruta) -> 21,35s (KD-tree) --
  **~2,7x mais rápido**. Bônus: essa mudança também corrigiu uma
  ineficiência que já tínhamos documentado no FCM (que também pagava pelo
  clustering não-utilizado da `opfython`) -- agora nem FCM nem
  `density_kdtree` pagam esse custo.

  **Confirmado no Thyroid real** (sigma=1.15, k_max=20): density=135.7s
  vs. density_kdtree=69.0s -- **~1.97x mais rápido**, com acurácia
  **idêntica** (0.7543 = 0.7543) entre os dois, confirmando o speedup
  exato também em dados reais, não só sintéticos.

  **Limitação documentada**: só funciona com `search_best_k=False` (k_max
  fixo) -- a busca de corte mínimo da `opfython` sobre uma faixa de k é
  um algoritmo diferente, não replicado aqui. 33 testes passando.
  `compare_membership_sources.py` atualizado para incluir as 3 fontes
  (`density`, `density_kdtree`, `fcm`) nas configs dos 8 datasets já
  existentes.

## Itens menores

- [x] **Erro de rotação de log no Windows** (`WinError 32`, aparece
  repetidamente nos runs longos) — não trava a execução, mas polui a saída;
  vale suprimir ou reconfigurar o logging dos scripts de experimento.
  **Resolvido**: `fuzzy_opf/__init__.py` desativa o file handler
  problemático da `opfython` antes de qualquer outro import.
- [x] **Teste de cobertura para dataset de uma única classe** — o
  fallback de `_find_prototypes` pra esse caso nunca foi testado
  explicitamente em `test_model.py`. **Resolvido**, teste adicionado.
