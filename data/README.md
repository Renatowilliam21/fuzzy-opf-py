# data/

- `raw/` — datasets originais em formato OPF (`.dat` binário do LibOPF, ou
  `.txt`/`.csv` já convertidos). **Não versionados no Git** (ver
  `.gitignore`), exceto o dataset sintético de teste rápido.

## Como obter os datasets do artigo

Os doze datasets usados no paper (Boat, Cone-Torus, Four-Class, Data1,
Data2, Data3, Thyroid, Breast Tissue, Landsat Satellite, MPEG-7 BAS,
Electric Industrial/Commercial Profiles) — os públicos estão no repositório
original da LibOPF: https://github.com/jppbsi/LibOPF/tree/master/data

Baixe o(s) `.dat` desejado(s) para `data/raw/`. O loader
(`fuzzy_opf.load_dataset`) converte `.dat` -> `.txt` automaticamente na
primeira leitura (usa `opfython.utils.converter.opf2txt`) e reaproveita o
`.txt` gerado nas próximas execuções.

## Dataset sintético (smoke test)

`python experiments/make_synthetic_dataset.py` gera
`data/raw/synthetic_quicktest.txt`, usado pela CI e pela config
`experiments/configs/quicktest.yaml` para validar o pipeline rapidamente
sem depender de dados externos.
