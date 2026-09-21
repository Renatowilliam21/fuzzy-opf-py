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

## Trabalhos relacionados (pesquisa de literatura, 2026-09-18)

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
- [ ] **2. AUC-ROC** -- métrica adicional para datasets desbalanceados
  (Thyroid especialmente), complementando a matriz de confusão que já
  fizemos. Barato, não precisa retreinar.
- [ ] **3. Baselines externos** (SVM com kernel RBF, Random Forest,
  XGBoost/LightGBM, k-NN fuzzy) -- hoje só comparamos Fuzzy-OPF vs. OPF
  padrão; nunca comparamos contra classificadores fora da família OPF.
  Mais trabalhoso (treinar modelos novos via scikit-learn nos datasets já
  validados), mas alto retorno para o artigo.
- [ ] **4. Teste de Friedman + post-hoc Nemenyi** -- depende do item 3
  (precisa de 3+ classificadores para fazer sentido; com só Fuzzy-OPF vs.
  OPF, Wilcoxon já basta).
- [ ] **5. Pertinência via Fuzzy C-Means (FCM) ou Gaussiana real** --
  diferente do que já fizemos (`membership_kind` muda só o *mapeamento*
  densidade->pertinência, Eq. 5); isso mudaria o *cálculo da densidade em
  si*. Extensão genuína, mais envolvida que o item de pertinência já
  fechado.
- [ ] **6. Formalizar teoricamente a quebra da propriedade "smooth" da
  função de custo** -- já temos evidência empírica/mecanística forte
  disso (o bug do ciclo infinito no heap, corrigido com a guarda
  `BLACK`), mas falta escrever a análise formal de sob quais condições
  matemáticas exatas a garantia de otimalidade global do OPF se mantém ou
  se perde com o produto `F_Theta(u) * max{C(q), d(q,u)}`. Isso é
  trabalho de redação/prova, não de código.
- [ ] **7. Subamostragem/aproximação de densidade (KD-trees, vizinhos
  aproximados)** para escalar o clustering em datasets maiores que o
  Thyroid (mais complicado, mais especulativo) -- complementa o que já
  fizemos (cache compartilhado, paralelismo, achado de que k_max pequeno
  já basta no Thyroid), mas ataca o problema por outro ângulo (reduzir o
  n em vez de reduzir k).

## Itens menores

- [x] **Erro de rotação de log no Windows** (`WinError 32`, aparece
  repetidamente nos runs longos) — não trava a execução, mas polui a saída;
  vale suprimir ou reconfigurar o logging dos scripts de experimento.
  **Resolvido**: `fuzzy_opf/__init__.py` desativa o file handler
  problemático da `opfython` antes de qualquer outro import.
- [x] **Teste de cobertura para dataset de uma única classe** — o
  fallback de `_find_prototypes` pra esse caso nunca foi testado
  explicitamente em `test_model.py`. **Resolvido**, teste adicionado.
