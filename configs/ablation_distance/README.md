# Ablação das variantes de DensGNN

Nove configurações: Cora, CiteSeer e PolBlogs, cada um com `adaptive`, `core`
e `border`. Cada configuração habilita apenas `densgnn` e solicita sete runs,
com as mesmas seeds **2030–2036**. Total deste lote: **63 novas execuções**.

O lote anterior usou as seeds 2026–2028. As novas execuções usam os mesmos
diretórios de resultados, com novas pastas de run para preservar os resultados
anteriores. Alterar estas configurações não inicia treinos.

| Dataset | Adaptive | Core | Border |
| --- | --- | --- | --- |
| Cora | [adaptive.yaml](cora/adaptive.yaml) | [core.yaml](cora/core.yaml) | [border.yaml](cora/border.yaml) |
| CiteSeer | [adaptive.yaml](citeseer/adaptive.yaml) | [core.yaml](citeseer/core.yaml) | [border.yaml](citeseer/border.yaml) |
| PolBlogs | [adaptive.yaml](polblogs/adaptive.yaml) | [core.yaml](polblogs/core.yaml) | [border.yaml](polblogs/border.yaml) |

## Comparação controlada

Dentro de cada dataset, somente `models.densgnn.params.point_selection` muda
entre as variantes, além do identificador da experiência. A distância de
suporte permanece `support_distance: mrd` nos três casos. Os demais parâmetros,
as métricas, a entrada de dados e as seeds são iguais.

Os parâmetros foram copiados dos respectivos `densgnn.yaml` em
`configs/core_border_adaptive_densgnn/core_border_adaptive_<dataset>/`.
Preservamos os valores específicos de cada dataset:

| Parâmetro | Cora | CiteSeer | PolBlogs |
| --- | ---: | ---: | ---: |
| `epochs` | 200 | 200 | 200 |
| `warmup_epochs` | 5 | 5 | 20 |
| `learning_rate` | 0.005 | 0.005 | 0.005 |
| `update_p_interval` | 1 | 1 | 5 |
| `hdbscan_min_cluster_size` | 5 | 5 | 10 |
| `hdbscan_min_samples` | 30 | 42 | 5 |
| `mrd_k` | 30 | 42 | 5 |

Mantidos `device: cuda`, `select_best_nmi: false`, as 12 métricas do benchmark,
a gravação de labels, embeddings, logs, metadados e figuras, e o limite existente
de **600 segundos por execução**, incluindo treino e avaliação. O limite pode
registrar uma tentativa como falha/OOT; sete runs significam sete tentativas,
sem repetição automática para substituir falhas.

## Saídas

Todos os resultados serão escritos em:

```text
results/ablation_distance/<dataset>_<variante>/
└── runs/<run_id>/
    ├── metrics.csv
    └── densgnn/
```

Exemplos: `results/ablation_distance/cora_core/` e
`results/ablation_distance/polblogs_adaptive/`. Os identificadores distintos
separam as nove combinações e não reutilizam os resultados do benchmark anterior.
O runner atual grava um CSV por run; as configurações não criam um CSV agregado
na raiz de `ablation_distance/`.

## Inspecionar e executar posteriormente

Validar uma configuração e visualizar o comando, sem treinar:

```bash
python scripts/main.py --config configs/ablation_distance/cora/core.yaml --model densgnn --dry-run
```

Quando os arquivos forem revisados, uma configuração poderá ser executada com:

```bash
python scripts/main.py --config configs/ablation_distance/cora/core.yaml --model densgnn
```

Para executar as nove configurações posteriormente, a partir da raiz:

```bash
for dataset in cora citeseer polblogs; do
  for variant in adaptive core border; do
    python scripts/main.py --config "configs/ablation_distance/$dataset/$variant.yaml" --model densgnn
  done
done
```

## Figura e análise estatística

Execute todas as células de
[`scripts/notebooks/ablation_distance_analysis.ipynb`](../../scripts/notebooks/ablation_distance_analysis.ipynb).
O notebook lê os resultados disponíveis, exibe quatro painéis (NMI, ARI, ACC
e F1-macro) com seeds individuais e média ± desvio padrão, e exporta a figura
em PNG/PDF/SVG para `results/ablation_distance/analysis/`.

Os boxplots são apresentados em uma figura independente por métrica, com os
datasets no eixo X e as cores indicando adaptive, core e border. Os arquivos
seguem `ablation_distance_boxplot_<metrica>.png`, `.pdf` e `.svg`, no mesmo
diretório. Para gerar somente NMI, configure `BOXPLOT_METRICS = ["nmi"]`;
o PDF correspondente é `ablation_distance_boxplot_nmi.pdf`.

A última seção do notebook também exporta o NMI em TikZ/PGFPlots:
`ablation_distance_boxplot_nmi_tikz.tex` para incluir no artigo e
`ablation_distance_boxplot_nmi_tikz_standalone.tex` para compilar separadamente.
Os quartis, bigodes e outliers são os mesmos do Matplotlib. O preâmbulo e o
exemplo de `\input` estão no notebook; a compilação do PDF é automática quando
`tectonic` ou `pdflatex` estiver disponível. Os números e seeds usados ficam
registrados nos arquivos `_tikz_stats.csv` e `_tikz_metadata.json`.
Por padrão, são usadas as seeds dos YAMLs. Para reunir os dois lotes somente
no boxplot, configure `BOXPLOT_SEEDS = [2026, 2027, 2028, *range(2030, 2037)]`
na primeira célula. Essa seleção não modifica os testes de Friedman por
permutação/Holm baseados nos YAMLs.

A seção **SciPy Friedman** chama `scipy.stats.friedmanchisquare` diretamente
para as mesmas seeds selecionadas no boxplot e imprime `statistic` e `pvalue`,
por dataset e entre datasets. Esses p-valores são assintóticos e sem ajuste
por múltiplos testes. Os valores completos ficam em
`results/ablation_distance/analysis/friedman_scipy_boxplot_runs.csv`.

As tabelas de Friedman incluem testes por dataset com seeds pareadas e testes
entre datasets usando as médias por grafo. A decisão usa o p-valor por
permutação (exato quando a enumeração cabe no limite configurado; Monte Carlo
nos demais casos) e correção de Holm em duas famílias
explicitadas no notebook. Testes incompletos ficam pendentes; reexecute o
notebook quando as runs terminarem. Nenhum treino é iniciado pela análise.

O notebook seleciona as seeds indicadas nos YAMLs. Com estas configurações,
ele considera somente as seeds 2030–2036; as runs antigas continuam salvas,
mas não entram automaticamente nas médias e testes deste lote.
