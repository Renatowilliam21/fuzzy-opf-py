# Fuzzy-OPF (Python)

Port do `LibOPF_fuzzy` (C) para Python, construído sobre as libs do Recogna
(`opfython` + `opytimizer`), implementando:

> R. W. R. de Souza, J. V. C. de Oliveira, L. A. Passos, W. Ding, J. P. Papa,
> V. H. C. de Albuquerque. *"A Novel Approach for Optimum-Path Forest
> Classification Using Fuzzy Logic."* IEEE Transactions on Fuzzy Systems, 2019.

## Estrutura do repositório

```
fuzzy-opf/
├── .github/workflows/ci.yml   # roda os testes + smoke test a cada push/PR
├── src/fuzzy_opf/             # o pacote (instalável via `pip install -e .`)
│   ├── model.py                #   classe FuzzyOPF (treino/predição)
│   ├── tuning.py                #   busca de hiperparâmetros via GA (opytimizer)
│   └── datasets.py               #   loader de datasets no formato OPF
├── experiments/                # scripts + configs para rodar experimentos
│   ├── configs/*.yaml
│   ├── run_experiment.py
│   └── make_synthetic_dataset.py
├── data/raw/                   # datasets (não versionados, ver data/README.md)
├── results/<dataset>/*.csv     # saída dos experimentos (não versionados)
├── notebooks/                  # análises exploratórias / gráficos
├── tests/                      # pytest
├── pyproject.toml
├── LICENSE (Apache-2.0)
└── CITATION.cff
```

Cada pasta relevante tem seu próprio `README.md` com detalhes
(`data/README.md`, `experiments/README.md`, `results/README.md`).

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate   # opcional, mas recomendado
pip install -e ".[dev]"
```

## Uso rápido

```python
from fuzzy_opf import FuzzyOPF, load_dataset

X, y = load_dataset("data/raw/boat.dat")
model = FuzzyOPF(k_max=20, sigma=0.6, search_best_k=True)
model.fit(X, y)
preds = model.predict(X)
```

## Rodando experimentos

```bash
# teste rápido, sem depender de dataset externo
python experiments/make_synthetic_dataset.py
python experiments/run_experiment.py --config experiments/configs/quicktest.yaml

# experimento "de verdade": copie experiments/configs/template.yaml,
# aponte para um dataset em data/raw/ e rode
python experiments/run_experiment.py --config experiments/configs/boat.yaml
```

## Testes

```bash
pytest -q
```

---

## Colocando isso no GitHub

Passo a passo a partir desta pasta (`fuzzy-opf/`):

```bash
git init
git add .
git commit -m "chore: scaffold Fuzzy-OPF Python port (model, tuning, experiments)"

# crie o repositório vazio no GitHub antes (via web ou `gh repo create`),
# depois aponte o remoto e envie:
git branch -M main
git remote add origin git@github.com:<seu-usuario>/fuzzy-opf.git
git push -u origin main
```

Se preferir criar o repositório direto pelo terminal (com o GitHub CLI
instalado e autenticado):

```bash
gh repo create fuzzy-opf --private --source=. --remote=origin --push
```

### Convenções sugeridas

- **Branches**: `main` sempre estável/rodável; features e experimentos em
  `feat/<nome-curto>` ou `exp/<dataset>-<o-que>` (ex.: `exp/thyroid-ga-vs-pso`),
  com PR para `main` quando o resultado for para o repositório principal.
- **Commits**: prefixo por tipo ajuda a navegar o histórico depois —
  `feat:`, `fix:`, `exp:` (resultado de experimento), `docs:`, `chore:`.
- **Tags**: quando uma rodada de experimentos sustentar uma submissão
  (ex.: ERCEMAPI, SBESC), marque o commit correspondente com uma tag
  (`git tag -a v0.1-ercemapi2026 -m "..."`) para conseguir voltar exatamente
  àquele estado do código depois.
- **CI**: `.github/workflows/ci.yml` já roda `pytest` e o smoke test do
  pipeline de experimentos em Python 3.11 e 3.12 a cada push/PR — qualquer
  quebra aparece antes de chegar no `main`.
- **Dados e resultados**: ficam fora do Git por padrão (`.gitignore`); veja
  `results/README.md` para como versionar pontualmente os números que forem
  para um artigo, sem acumular todo o histórico de execuções no repositório.

## O que muda em relação ao código C original

| Ponto levantado na análise do C | Como foi endereçado no port |
|---|---|
| Releitura do dataset do disco 2x (`ReadSubgraph` duas vezes) | Um único array NumPy em memória, reaproveitado |
| Pertinência aplicada ao nó **fonte** (`p`) em vez do nó **candidato** (`u`), divergindo da Eq. 6/Algoritmo 3 do artigo | `membership_side="target"` (padrão) implementa a equação do artigo; `membership_side="source"` reproduz o comportamento legado do C, para comparação (ver `tests/test_model.py`) |
| `sigma` sem validação de faixa; divisão por zero se `rho_max == rho_min` | `ValueError` fora de `(0, 1.5]`; guarda explícita para densidade degenerada |
| Grid search força-bruta 16×6 = 96 retrains, reclusterizando a cada combinação de `sigma` | Busca por metaheurística (GA) + cache de clustering por `k_max`, evitando reclusterizar quando o mesmo `k_max` reaparece |
| `classifier.opf` sobrescrito a cada iteração do grid search | Cada run vira uma linha em `results/<dataset>/<timestamp>.csv`, nada é jogado fora silenciosamente |
| Sem testes, warnings de compilação (buffer overflow em `kFoldSubgraph`, variáveis não usadas) | `tests/test_model.py` (pytest) + CI a cada push |

## Próximos passos sugeridos

1. **Validar a Eq. 6**: rodar `membership_side="target"` vs `"source"` nos
   datasets do artigo e comparar com a Tabela I publicada.
2. **Comparar GA vs. outras meta-heurísticas do Opytimizer** (PSO, DE...) —
   trocar o import em `tuning.py` já é suficiente, dada a interface comum
   do Opytimizer; um script `experiments/run_hyperparam_search.py` para
   automatizar essa comparação é o próximo candidato natural.
3. Paralelizar a avaliação dos agentes do GA (`joblib.Parallel`), já que
   cada indivíduo é independente.
4. Publicar como PR para a própria `opfython`, que hoje só tem
   `SupervisedOPF`, `KNNSupervisedOPF`, `SemiSupervisedOPF` e
   `UnsupervisedOPF` — o Fuzzy-OPF preencheria essa lacuna.
