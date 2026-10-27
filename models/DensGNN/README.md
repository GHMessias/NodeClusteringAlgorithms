# Density_Deep_Clustering_gnn

Framework de benchmark para algoritmos de clustering em grafos, com execução guiada por YAML e organização inspirada no GraphGym, mas sem depender da biblioteca.

## Objetivo

Este repositório serve para comparar algoritmos de clustering em grafos sob uma interface unificada:

- datasets carregados via `torch_geometric.datasets`;
- execução centralizada por `main.py`;
- seleção do algoritmo por registry;
- configs em `.yaml`;
- saída padronizada em `results/`.

O foco atual está em benchmarks de clustering de nós em grafos atribuídos, começando por `Cora`.

## Encoders do DensGNN

`models.densgnn.params.encoder_type` aceita `gcn` (padrão original), `gat`
ou `sage` (SAGEConv/GraphSAGE com agregação média). Todos usam duas camadas,
com ReLU e dropout de features entre elas. GAT usa uma cabeça por camada,
sem concatenação nem dropout de atenção, como no encoder existente de
DeepDensGNN. O decoder GAE, a reconstrução, a loss de clustering, a seleção
de suportes e o cálculo de MRD mantêm suas definições existentes.

O encoder é registrado em `metrics.json` e `metadata.yaml`. Configurações sem
`encoder_type` continuam construindo a mesma classe `TwoLayerGCNEncoder`.
O sweep de min_pts e encoders fica em `configs/ablation_minpts_encoder/`.

## Seleção de suportes do DensGNN

O parâmetro `algorithm.params.point_selection` aceita três estratégias:

- `core`: usa pontos com força de pertencimento ao cluster acima do threshold;
- `border`: usa pontos não classificados como ruído abaixo do threshold;
- `adaptive`: combina as distâncias aos conjuntos core e border com um peso
  específico para cada cluster. O alias `adaptative` também é aceito.

No modo adaptativo, a mediana das core distances estima o raio local de cada
cluster. A densidade relativa é normalizada de maneira robusta por MAD (com
fallback para IQR e desvio-padrão) e transformada em `alpha_j` pela função
sigmoide. A distância usada pela loss é:

```text
d(i, j) = (1 - alpha_j) * d_core(i, j) + alpha_j * d_border(i, j)
```

Clusters relativamente esparsos recebem mais peso core; clusters relativamente
densos recebem mais peso border. O cálculo usa somente embeddings e resultados
do HDBSCAN, sem consultar os rótulos verdadeiros. As métricas
`selection_alpha_mean`, `selection_alpha_min` e `selection_alpha_max`, além dos
valores por cluster no metadata, permitem auditar a decisão.

## Ablação da distância aos pontos de suporte

`algorithm.params.support_distance` aceita `mrd` (padrão, comportamento
original) ou `euclidean`. A alternativa euclidiana substitui apenas a MRD
pela distância euclidiana entre o embedding do nó e cada ponto de suporte;
os clusters do HDBSCAN, a seleção `core`/`border`/`adaptive`, os pesos dos
suportes, a transformação em atribuições suaves e a loss permanecem iguais.
Não se trata de KMeans nem de distância ao centroide. Por exemplo:

```yaml
models:
  densgnn:
    params:
      support_distance: euclidean
```

No modo `adaptive`, `mrd_k` continua sendo usado para estimar a densidade
relativa que determina `alpha_j`, mesmo com `support_distance: euclidean`.
O valor efetivo da opção é registrado em `metrics.json` e no metadata da run.

Para salvar uma curva de NMI sem expor os rótulos ao treinamento, habilite:

```yaml
models:
  densgnn:
    params:
      save_epoch_nmi: true
      evaluation_interval: 1
```

O modelo registra por época as métricas internas tanto no embedding aprendido
(`*_embedding`) quanto nos atributos originais dos nós (`*_features`). Depois
que o último epoch termina, o adaptador compara as atribuições capturadas com
`data.y` para acrescentar o NMI e grava tudo em
`algorithm.evaluation_history` no `metrics.json`. Os campos disponíveis são:

- `nmi`;
- `silhouette_embedding` e `silhouette_features`;
- `calinski_harabasz_embedding` e `calinski_harabasz_features`;
- `davies_bouldin_embedding` e `davies_bouldin_features`;
- `num_clusters` e `noise_ratio`.

Esse histórico nunca é usado na loss, em early stopping ou na escolha do
resultado final. As métricas internas, principalmente silhouette, aumentam o
tempo de execução quando calculadas a cada época; um `evaluation_interval`
maior reduz essa frequência.

O estado final também inclui `algorithm.cluster_support_stats` no
`metrics.json`. Para cada cluster do HDBSCAN são registrados o tamanho do
cluster, as quantidades candidatas e efetivas de core/border, o total de nós de
suporte distintos e, no modo adaptive, `alpha_j` e a densidade relativa. O
runner central copia essas estatísticas para `cluster_support_stats` no
`metadata.yaml` da execução.

O histórico de treinamento é salvo como `algorithm.loss_history` no
`metrics.json` e como `loss_history` no `metadata.yaml`. Cada época contém
`reconstruction_loss`, `clustering_loss` e `total_loss`, sendo a última
calculada como `reconstruction_loss + clustering_loss_gamma *
clustering_loss`. Esses valores são somente diagnósticos e não modificam a
seleção do resultado final.

## Algoritmos Implementados

Hoje o framework já suporta:

- `kmeans_features`: `KMeans` aplicado diretamente nas features dos nós, ignorando topologia.
- `gae_kmeans_embeddings`: treino de `GAE` para obter embeddings latentes e depois `KMeans`.
- `ollivier_ricci_community`: clustering topológico com `GraphRicciCurvature` usando `ricci_community()`.
- `dgcss`: implementação própria inspirada no paper/repositório DGCSS, com encoder atencional, seed selection e loss de clustering via `KL`.

## Estratégias de Sementes do DGCSS

O algoritmo `dgcss` já aceita as seguintes estratégias registradas para seleção de sementes:

- `betweenness_centrality`
- `closeness_centrality`
- `kmeans`
- `random`

Essas estratégias vivem em `src/graph_benchmark/seed_selection/` e podem ser trocadas diretamente no YAML ou com `--set algorithm.params.seed_selector=...`.

## Organização

```text
.
├── configs/
│   ├── cora_dgcss_bc.yaml
│   ├── cora_gae_kmeans.yaml
│   ├── cora_kmeans_features.yaml
│   └── cora_ollivier_ricci_community.yaml
├── data/
│   └── raw/
├── DGCSS/
├── main.py
├── results/
├── scripts/
│   ├── run_cora_dgcss.py
│   ├── run_cora_gae_kmeans.py
│   ├── run_cora_kmeans.py
│   └── run_cora_ollivier_ricci_community.py
└── src/
    └── graph_benchmark/
        ├── clustering/
        ├── config/
        ├── datasets/
        ├── evaluation/
        ├── experiments/
        ├── models/
        ├── seed_selection/
        ├── utils/
        ├── register.py
        ├── registry.py
        └── runner.py
```

Resumo dos diretórios:

- `configs/`: cenários de benchmark em YAML.
- `data/raw/`: datasets baixados pelo `torch_geometric`.
- `results/`: métricas e atribuições de cluster geradas pelos experimentos.
- `scripts/`: wrappers opcionais para execuções rápidas.
- `src/graph_benchmark/clustering/`: algoritmos registrados.
- `src/graph_benchmark/models/`: componentes de modelagem, como `GAE` e `DGCSS`.
- `src/graph_benchmark/seed_selection/`: estratégias de seed selection do `DGCSS`.
- `src/graph_benchmark/experiments/`: pipelines reutilizáveis de execução.
- `DGCSS/`: repositório de referência clonado localmente para estudo; o framework principal não depende diretamente dele.

## Como o Framework Roda

O fluxo padrão é:

1. `main.py` lê um arquivo YAML.
2. `run.experiment` seleciona o pipeline registrado.
3. `dataset.loader` seleciona o carregador do dataset.
4. `algorithm.name` escolhe o algoritmo de clustering.
5. o experimento executa o algoritmo e salva os artefatos em `results/`.

Hoje o pipeline principal é `node_feature_clustering`, que é usado tanto para os baselines simples quanto para `GAE`, `Ricci` e `DGCSS`.

## Instalação

O ambiente precisa ter pelo menos:

- `torch`
- `torch_geometric`
- `scikit-learn`
- `pyyaml`
- `scipy`
- `networkx`

Para a baseline de Ricci, também é necessário:

- `GraphRicciCurvature`

Exemplo:

```bash
./venv/bin/pip install scikit-learn pyyaml scipy networkx GraphRicciCurvature
```

## Execução

### Entrada principal

```bash
./venv/bin/python main.py --cfg <arquivo.yaml>
```

### Baselines disponíveis

`KMeans` nas features:

```bash
./venv/bin/python main.py --cfg configs/cora_kmeans_features.yaml
```

`GAE + KMeans`:

```bash
./venv/bin/python main.py --cfg configs/cora_gae_kmeans.yaml
```

`Ollivier-Ricci community`:

```bash
./venv/bin/python main.py --cfg configs/cora_ollivier_ricci_community.yaml
```

`DGCSS` com seed selection por Betweenness Centrality:

```bash
./venv/bin/python main.py --cfg configs/cora_dgcss_bc.yaml
```

### Variando o seed selector do DGCSS

Trocar para `closeness_centrality`:

```bash
./venv/bin/python main.py \
  --cfg configs/cora_dgcss_bc.yaml \
  --set algorithm.params.seed_selector=closeness_centrality output.dir=results/cora/dgcss_cc
```

Trocar para `kmeans`:

```bash
./venv/bin/python main.py \
  --cfg configs/cora_dgcss_bc.yaml \
  --set algorithm.params.seed_selector=kmeans output.dir=results/cora/dgcss_kmeans
```

Trocar para `random`:

```bash
./venv/bin/python main.py \
  --cfg configs/cora_dgcss_bc.yaml \
  --set algorithm.params.seed_selector=random output.dir=results/cora/dgcss_random
```

### Overrides úteis

Mudar seed global e diretório de saída:

```bash
./venv/bin/python main.py \
  --cfg configs/cora_gae_kmeans.yaml \
  --set run.seed=7 output.dir=results/cora/gae_kmeans_seed7
```

Rodar um smoke test curto do DGCSS:

```bash
./venv/bin/python main.py \
  --cfg configs/cora_dgcss_bc.yaml \
  --set algorithm.params.epochs=5 output.dir=results/cora/dgcss_smoke
```

## Saídas Geradas

Cada execução salva pelo menos:

- `metrics.json`
- `assignments.csv`

Exemplos de diretórios:

- `results/cora/kmeans_features/`
- `results/cora/gae_kmeans/`
- `results/cora/ollivier_ricci_community/`
- `results/cora/dgcss_bc/`

## Estrutura dos YAMLs

Os configs seguem esta ideia:

```yaml
run:
  experiment: node_feature_clustering
  seed: 42

dataset:
  loader: planetoid
  name: Cora
  root: data/raw

algorithm:
  name: dgcss
  params:
    seed_selector: betweenness_centrality
    epochs: 400
    ...

output:
  dir: results/cora/dgcss_bc
  save_assignments: true
```

## Estado Atual do DGCSS no Framework

A implementação do `dgcss` no framework é uma versão própria inspirada no paper e no repositório clonado em `DGCSS/`, mas integrada ao padrão do projeto atual:

- encoder `GAT` de duas camadas;
- matriz de transição multi-hop;
- seleção de sementes por registry;
- perda de clustering com `KL`;
- loss total de reconstrução + clustering;
- avaliação integrada com `NMI`, `ARI`, `Purity`, `Accuracy`, `Modularity` e, quando fizer sentido, `Silhouette`.

## Observação Sobre o Diretório `DGCSS/`

O diretório `DGCSS/` foi clonado como referência para estudo do paper e da implementação original. Ele ajuda a:

- entender a arquitetura do método;
- comparar a lógica do paper com a implementação original;
- inspirar novas integrações no framework.

Mas os benchmarks do framework principal devem ser executados pelo `main.py` na raiz do projeto, não pelo `run.py` do repositório clonado.

## Próximos Passos Naturais

- adicionar novos datasets além de `Cora`;
- criar YAMLs prontos para todas as variantes do `DGCSS`;
- incluir novos seed selectors topológicos;
- adicionar outras famílias de métodos de clustering em grafos;
- versionar benchmarks e consolidar tabelas comparativas automaticamente.
