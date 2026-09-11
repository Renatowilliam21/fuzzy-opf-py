# experiments/

Scripts e configs para reproduzir o protocolo experimental do artigo
(20 execuções, split 60/20/20, hiperparâmetros ajustados no split de
validação) usando a versão Python do Fuzzy-OPF.

## Fluxo

1. Coloque o dataset em `data/raw/` (veja `data/README.md`).
2. Copie `experiments/configs/template.yaml` para
   `experiments/configs/<dataset>.yaml` e ajuste `dataset`, `n_runs` e o
   bloco `tuning`.
3. Rode:

   ```bash
   python experiments/run_experiment.py --config experiments/configs/<dataset>.yaml
   ```

4. O resultado (uma linha por método por execução) é salvo em
   `results/<dataset>/<timestamp>.csv` e um resumo (média ± desvio padrão)
   é impresso no terminal.

## Teste rápido (sem dataset externo)

```bash
python experiments/make_synthetic_dataset.py
python experiments/run_experiment.py --config experiments/configs/quicktest.yaml
```

## Arquivos

| Arquivo | Papel |
|---|---|
| `run_experiment.py` | Roda o protocolo completo (OPF padrão vs. Fuzzy-OPF) para um dataset/config |
| `make_synthetic_dataset.py` | Gera um dataset sintético pequeno para testes rápidos / CI |
| `configs/template.yaml` | Modelo de configuração, comentado |
| `configs/quicktest.yaml` | Config mínima usada pela CI |

## Próximos scripts (sugeridos, ainda não implementados)

- `run_hyperparam_search.py`: comparar GA vs. outras meta-heurísticas do
  Opytimizer (PSO, DE, ...) no mesmo orçamento de avaliações, e comparar
  contra o grid search original do `fuzzy_validation_opf.c` (tempo total e
  acurácia final) — ver README raiz, seção "Próximos passos".
