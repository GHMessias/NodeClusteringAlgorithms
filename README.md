# Node Clustering Algorithms Benchmark

This repository provides a configuration-driven benchmark for graph node-clustering baselines. The orchestration layer standardizes configuration, execution logs, predicted labels, evaluation metrics, and run metadata without changing the baselines' clustering objectives.

## Models

The benchmark exposes the following twenty entries:

- `kmeans_attributes`
- `node2vec_kmeans`
- `gae_kmeans`
- `vgae_kmeans`
- `arga_kmeans`
- `coane`
- `coane_topk`
- `dgc_efr`
- `soli`
- `ucode`
- `ns4gc`
- `dgcluster`
- `dgcss`
- `sanece`
- `egc_vgae`
- `dsagc`
- `scgc`
- `densgnn`
- `deep_densgnn`
- `dmon`

`coane` and `coane_topk` are separate benchmark entries. DGC-EFR is executed as a three-stage pipeline: AE pretraining, GAE pretraining, and final clustering training.
`kmeans_attributes` is a non-neural baseline that applies KMeans directly to
`data.x`; it deliberately ignores graph edges. `node2vec_kmeans` learns a
structural representation with `torch_geometric.nn.Node2Vec` and the installed
`torch_cluster` backend, then applies KMeans; it deliberately ignores node
attributes. Both save the exact representation used for clustering as
`embeddings.npy`.
`gae_kmeans`, `vgae_kmeans`, and `arga_kmeans` use matched two-layer GCN
encoders and cluster their final node embeddings with KMeans. GAE optimizes
edge reconstruction, VGAE adds a node-normalized KL term, and ARGA adds an
adversarial discriminator against a Gaussian latent prior. All three save the
learned representation, loss history, KMeans centers, and final assignments in
their isolated model result directory.
DGCSS uses the KMeans centroid-selection variant and exports its final-epoch
assignments and GAT embeddings through the common benchmark interface.
DensGNN uses the same configured `data_pt`, seed, device, result directory, and centralized evaluator as the other entries.
For a controlled support-distance ablation, set `models.densgnn.params.support_distance`
to `mrd` (default) or `euclidean`. Both modes keep HDBSCAN and the same
core/border/adaptive support-selection logic; only the node-to-support distance
used in the clustering loss changes.
`models.deep_densgnn.params.support_distance` accepts the same values. In its
`euclidean` mode, the learned MRD scale weights are inactive and reported as
such in the run metadata.
Its benchmark result is produced from the final training epoch; ground-truth labels are not used for checkpoint selection.
`deep_densgnn` is an independent experimental variant with learned, cluster-specific
mixtures of core-distance scales. Set `models.deep_densgnn.params.encoder_type`
to `gcn` (default), `gat`, or `sage` (GraphSAGE) to choose its two-layer encoder.
It leaves the original DensGNN unchanged;
see [its model notes](models/DeepDensGNN/README.md) for the MRD definition,
configuration, diagnostics, and limitations.
SOLI is always executed on CPU, independently of `experiment.device`, to keep its mixed PyTorch/TensorFlow pipeline isolated from CUDA.
DMoN uses PyTorch Geometric's `DMoNPooling` operator with an unsupervised GCN-based node-clustering adapter. Its `n_clusters` parameter can be configured explicitly or inferred from `data.y` when set to `null`.
Because the PyG operator consumes a dense adjacency matrix, the DMoN entry has quadratic adjacency-memory usage and is intended for graphs that fit this representation.

## Install

Install the shared environment with:

```bash
pip install -r requirements.txt
```

Some original baseline paths have optional dependencies documented at the end of `requirements.txt`; they are not required by the standard `data.pt` benchmark flow.

The optional Newman-Riolo topology-based cluster-count estimator uses its
original C implementation and requires GCC plus the GNU Scientific Library
development files. On openSUSE, install them with:

```bash
sudo zypper install gsl-devel
```

## Input data

The configured `data_pt` file must be a PyTorch/PyG-style object containing:

```text
data.x           node features
data.edge_index  graph edges with shape [2, num_edges]
data.y           optional ground-truth labels
```

The runner validates that the final predicted-label count equals the graph's node count. If `data.y` is absent, supervised metrics are recorded as `NaN`, while metrics such as modularity and number of clusters are still computed. Fixed-K baselines currently require their cluster-count parameter to be configured explicitly in this case; no placeholder labels are fabricated.

For overlapping or multilabel datasets, set
`experiment.ignore_ground_truth: true` and request only label-independent
metrics. The runner preserves the source dataset and materializes a derived
`data_without_ground_truth.pt` under `results/<dataset>/inputs/`; this same
label-free graph is passed to every model and to the common evaluator. Use a
topology-based cluster-count policy such as `newman-riolo` when `K` cannot be
read from a single-label target.

### PPI benchmark graphs

PyG's PPI files contain several independent graphs and 121-dimensional overlapping labels, so they must not be passed directly to the single-graph runner. The two test graphs are materialized independently in:

```text
datasets/PPI/benchmark/test_graph_00.pt
datasets/PPI/benchmark/test_graph_01.pt
```

Regenerate them from `datasets/PPI/processed/test.pt` with:

```bash
python scripts/prepare_ppi.py --split test --graph-indices 0 1 --overwrite
```

The original targets are preserved as `data.y_multilabel`; `data.y` is intentionally absent because ACC, ordinary NMI, ARI, and Hungarian-matched F1 assume one class per node. The supplied PPI configurations therefore request only runtime, modularity, number of clusters, and embedding-space internal metrics:

```bash
python scripts/main.py \
  --config configs/ppi/ppi_test_graph_00.yaml \
  --model densgnn

python scripts/main.py \
  --config configs/ppi/ppi_test_graph_01.yaml \
  --model densgnn
```

Or run both graph configurations sequentially:

```bash
bash scripts/run.sh configs/ppi densgnn
```

The source mapping and graph statistics are recorded in `datasets/PPI/benchmark/manifest.yaml`. The small PPI YAMLs enable DensGNN for direct native-cluster experiments. Fixed-K competitors can accept the same label-free files when `cluster_count.method` is set to `newman-riolo`; their internal supervised reports are skipped and the common evaluator records external metrics as `NaN`.

## Configure

The central configuration is:

```text
configs/global_config.yaml
```

It contains four required sections and one optional section:

- `experiment`: dataset, input path, output root, device, seed, run count, and failure policy;
- `cluster_count`: optional common policy for determining K;
- `output`: per-run CSV name, retained artifacts, and automatic figure generation;
- `metrics.include`: metric columns to compute and append;
- `models`: enable/disable state and model-specific parameters recovered from the baseline defaults.

The optional top-level `cluster_count` section selects the common source of
the cluster count used by every fixed-K model. With `method: null`, the runner
counts the distinct labels in `data.y`. With `method: newman-riolo`, it runs the
topology-only Riolo-Cantwell-Reinert-Newman estimator once before the models and
uses the posterior mode of K after burn-in:

```yaml
cluster_count:
  method: newman-riolo  # or null to use data.y
  newman_riolo:
    burn_in: 1000
    timeout_seconds: null
```

The upstream estimator remains unchanged, including its 10,000 sweeps,
hard-coded maximum of 25 groups, and clock-based random seed. Its complete
trace, posterior table, diagnostics, and selected K are stored under
`results/<dataset>/cluster_count/<estimation_id>/`. DensGNN retains its native
HDBSCAN cluster count and does not receive the selected fixed K.

Paths in the configuration are resolved relative to the repository root, even when the command is launched from another working directory. Model parameters belong under `models.<model>.params`; the root CLI intentionally exposes only common experiment overrides.

To select the final benchmark result by the highest NMI observed during training
for every enabled model, set:

```yaml
experiment:
  select_best_nmi: true
```

The default is `false`, which preserves each baseline's original result selection.
When enabled, all training epochs (or clustering iterations for SOLI) report their
available hard assignments; methods that cluster embeddings run their existing
KMeans/Birch step for each candidate. DensGNN methods start reporting assignments
when the clustering phase becomes available, and also evaluate final inference.
DGC-EFR selects only from its final clustering stage; its pretraining stages are
unchanged. KMeans on attributes has just one result. The native final output is
also considered, and ties retain the first candidate observed.

This is **supervised result selection** using `data.y`, even though the losses
remain unchanged. Without ground truth (including `ignore_ground_truth: true`),
the runner warns and falls back to native selection. Additional clustering and
evaluation can increase runtime. This option works independently of
`metrics.include` and `save_epoch_nmi`.

The selected labels and corresponding embeddings become `labels.npy` and
`embeddings.npy`, and all metrics in the benchmark CSV are recomputed from that
pair. `metadata.yaml` records `result_selection`, including the NMI, epoch, and
stage. `selection/history.csv` records candidates and `selection/selection.json`
records the winner. Epochs are numbered from 1; `native_output` may have no epoch
when the baseline does not expose it. These are result artifacts, not resumable
weight checkpoints. Native baseline logs and auxiliary artifacts retain their
original meaning and may describe the baseline's own final output.

When `experiment.runs` is greater than one, `experiment.seed` is the initial seed. Seeds increase by one for each repetition; for example, `seed: 2026` with `runs: 3` produces seeds `2026`, `2027`, and `2028`. Each effective seed is recorded in its run ID, model metadata, and CSV rows.

For the CiteSeer GraphSAGE ablation, use
[`configs/configs_encodertype_citeseer/`](configs/configs_encodertype_citeseer/README.md).
It enables only `deep_densgnn` with `encoder_type: sage`, sweeps
`mrd_k = hdbscan_min_samples` from 2 to 60 in steps of 2, and uses three seeds
per configuration (90 executions). Run it with:

```bash
bash scripts/run.sh configs/configs_encodertype_citeseer deep_densgnn
```

Embeddings and metrics are saved under `results/encodertype_citeseer/runs/`.

## Run one model

```bash
python scripts/main.py \
  --config configs/global_config.yaml \
  --model ucode
```

The model must have `enabled: true` in the configuration. A disabled model requested explicitly produces a clear configuration error.

## Run all enabled models

```bash
bash scripts/run.sh
```

An alternative configuration can be supplied as the first argument:

```bash
bash scripts/run.sh configs/global_config.yaml
```

This delegates to:

```bash
python scripts/main.py --config configs/global_config.yaml --model all
```

Only entries with `enabled: true` are selected by `--model all`.

### Analyze runs from multiple result directories

Open [scripts/result_paths_analysis.ipynb](scripts/result_paths_analysis.ipynb)
and set `RESULT_PATHS`, for example:

```python
RESULT_PATHS = ["results/cora_adaptive", "results/cora_border", "results/cora_core"]
```

Run all cells to summarize successful runs separately for each path, dataset,
and model, with means, sample standard deviations, valid-value counts, and
comparison plots. By default, repeated seeds keep the latest successful attempt;
set `DEDUPLICATE_RETRIES = False` to include every successful run. Metrics are
discovered from `runs/*/metrics.csv`; `METRICS` can select a subset. CSV, PNG,
and LaTeX exports go to `results/result_paths_analysis/` by default.

### DensGNN warm-up comparison

Open [scripts/border_core_plot.ipynb](scripts/border_core_plot.ipynb) to train
DensGNN and compare its warm-up and final embeddings. The notebook recovers
hyperparameters from an existing model run's `metadata.yaml` and selects
`POINT_SELECTION = "border"` for clustering supports. `e_warm` is the
number of reconstruction-only epochs; `epochs` is the total including warm-up.
Both snapshots are computed in evaluation mode after their optimizer steps.

Training can also be run from the command line:

```bash
python scripts/train_warmup_comparison.py \
  --source-dir results/cora_50_runs/runs/20260921_193005_512984_seed2038_run13/densgnn \
  --e-warm 5 --epochs 200 --point-selection border \
  --point-probability-threshold 0.2
```

This creates a new directory under `results/<dataset>/runs/` containing
`embeddings_warmup.npy`, final `embeddings.npy`, `warmup_core_border.csv`,
`final_core_border.csv`, node IDs, final assignments, loss history and metadata.
The CLI defaults to the original seed, device and support selection; use
`--point-selection border` to override an adaptive run and `--device cpu` when
needed. The notebook sets `POINT_PROBABILITY_THRESHOLD = 0.2`; the CLI option
`--point-probability-threshold` overrides the same value in training and CSV
classification. Lowering the threshold reduces border candidates and increases
core candidates for fixed HDBSCAN results. New comparison run IDs include the
selected support mode. The notebook
additionally saves a joint t-SNE comparison as PNG/PDF
and CSVs with the projected coordinates. Core/border classification is computed
on each stage's original embeddings using the existing figure definition.

### Automatic figures

Set the following option to generate the analysis figures after every completed
run (including runs started with `--model all`):

```yaml
output:
  save_labels: true
  save_embeddings: true
  save_metadata: true
  generate_figures: true
```

The optional settings `figure_dpi`, `figure_max_points`, `figure_perplexity`,
and `figure_tsne_iterations` control image resolution, t-SNE sampling, and t-SNE
parameters. Figure generation is isolated from model execution: a plotting
error is recorded in the figure manifest and reported as a warning, but does
not change a successful model's benchmark status.

## Common overrides

The configuration remains the normal source of truth, but these temporary overrides are available for debugging and smoke tests:

```text
--dataset NAME
--data-pt PATH
--device cpu|cuda
--seed INTEGER
--runs INTEGER
```

To validate configuration and command construction for one or every enabled model without running training:

```bash
python scripts/main.py \
  --config configs/global_config.yaml \
  --model all \
  --dry-run
```

## Results

Every benchmark repetition receives a unique run ID. Its model metrics are aggregated in:

```text
results/<dataset>/runs/<run_id>/metrics.csv
```

Each model is isolated below the same run directory:

```text
results/<dataset>/runs/<run_id>/
├── metrics.csv
├── figures/
│   ├── benchmark_quality.png
│   ├── benchmark_internal.png
│   ├── benchmark_runtime_clusters.png
│   ├── benchmark_ranking.png
│   └── manifest.yaml
└── <model>/
    ├── labels.npy
    ├── embeddings.npy
    ├── stdout.log
    ├── stderr.log
    ├── metadata.yaml
    └── figures/
        └── embedding_projections.png
```

The run-level images compare all successful models in the run. Every model
with a saved representation receives a t-SNE overview comparing its
predicted partition with the ground truth when labels are available. DensGNN
also receives `epoch_metrics.png`, `loss_history.png`, and
`support_points.png` whenever the corresponding history is present.

Figures can also be generated or regenerated for an existing run:

```bash
python scripts/generate_figures.py \
  --run-dir results/cora/runs/<run_id> \
  --data-pt datasets/Cora/processed/data.pt
```

Learned node representations are retained by default. The runner normalizes every
available model representation to `embeddings.npy`, validates that it is a numeric
two-dimensional matrix with one row per node, and records its shape and dtype in
`metadata.yaml`. This behavior is controlled by `output.save_embeddings` and
defaults to `true` when the option is omitted.

Baseline-specific artifacts, such as DGC-EFR pretraining weights, stay inside the same model directory. A failure records `status=failed`, runtime, and an error message without deleting successful results from other models or earlier runs. Starting another run creates another isolated CSV instead of modifying metrics from a previous run.

Identification columns are always written to the run CSV:

```text
dataset,run_id,model,seed,status,error
```

`n_clusters` records the number of clusters actually produced by the model.

Metric columns follow `metrics.include`. Supported metrics are:

```text
runtime_seconds
acc
nmi
ari
f1_macro
f1_micro
modularity
n_clusters
silhouette
calinski_harabasz
davies_bouldin
dbcv
```

ACC and F1 use optimal Hungarian label matching. NMI and ARI compare the raw cluster assignments. Modularity uses the undirected graph represented by `data.edge_index` and does not require ground-truth labels.
Silhouette, Calinski-Harabasz, and Davies-Bouldin evaluate the final partition in the learned embedding space saved by each model as `embeddings.npy`; they do not require ground-truth labels. Silhouette and Calinski-Harabasz are better when higher, while Davies-Bouldin is better when lower. These fields are recorded as `NaN` when the number of clusters makes the metric undefined. A missing or incompatible embedding is reported as a failed model execution instead of silently falling back to the original `data.x` attributes.

Add `dbcv` to `metrics.include` to compute Density-Based Clustering Validation
on the same saved embeddings, using Euclidean distance and the MST/separation
primitives from [`hdbscan.validity`](https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.validity.validity_index).
All-points core distances use an algebraically equivalent scaled formula to
avoid overflow/underflow from inverse powers in high-dimensional embeddings.
DBCV does not require ground-truth labels; its range is [-1, 1], with higher
values indicating better clustering. Label `-1` denotes noise: noise contributes
no cluster score but remains in the total sample count used for weighting.
Cluster IDs are normalized internally without changing saved labels. Fewer than
two non-noise clusters, any singleton cluster, or non-finite scores produce `NaN`.
The `hdbscan` dependency is loaded only when DBCV is requested. Exact DBCV uses
pairwise distances within clusters and can require substantial time and memory
for large clusters; evaluation uses all nodes without subsampling.

## Manual evaluation

The common evaluator can also evaluate an existing labels file:

```bash
python scripts/evaluate.py \
  --data-pt datasets/custom/scatt_graph.pt \
  --labels path/to/labels.npy \
  --embeddings path/to/embeddings.npy
```

The evaluator and benchmark runner share the same metric implementation.

To evaluate only DBCV for an existing run, pass the same files with `--metrics dbcv`.

To add DBCV to existing `results/*_final/runs/*/metrics.csv` files without
training models again:

```bash
python scripts/update_dbcv.py
```

The updater uses saved `labels.npy` and `embeddings.npy`, preserves existing
columns and runtimes, and backs up each CSV as `metrics.csv.before_dbcv.bak`.
Failed model runs receive `NaN`. Each evaluated model receives a
`dbcv_evaluation.json` with the score, evaluation time, and reason for an
undefined score. The combined report is `results/dbcv_update_report.json`.
Repeated invocations reuse evaluations when the artifacts and evaluator have
not changed. Use `--results-root /path/to/results` for another result directory.
