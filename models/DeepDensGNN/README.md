# DeepDensGNN

Experimental variant of the existing DensGNN. By default it keeps the same GCN-GAE,
warm-up, reconstruction loss, HDBSCAN refresh, support selection, Q/P and KL
clustering loss. Only the single-k MRD is replaced with a cluster-conditioned
mixture of core distances. The original `models/DensGNN/` is unchanged.

Set `models.deep_densgnn.params.encoder_type` in the benchmark YAML to select
`gcn` (default), `gat`, or `sage` (GraphSAGE). Existing configs without this
key retain the original GCN encoder. For example:

```yaml
models:
  deep_densgnn:
    enabled: true
    params:
      encoder_type: gat
      encoder_hidden_channels: 128
      embedding_channels: 32
      dropout: 0.0
```

All three encoders use two convolution layers with ReLU and feature dropout
between them, and produce `embedding_channels` features per node. GAT uses
one attention head per layer, without attention dropout; GraphSAGE uses mean
aggregation. The GAE decoder, training losses and clustering procedure are
unchanged. The selected encoder is recorded in `metrics.json` and
`metadata.yaml`. Invalid encoder types are rejected.

`candidate_k: [3, 5, 8, 12, 16, 24]` defines the neighbor ranks. A single
top-k search up to the largest candidate supplies every core distance. Each
current HDBSCAN cluster has trainable zero-initialized logits; softmax of the
logits divided by `density_temperature` yields its mixture `pi`. The
diagnostic `k_eff` is the expectation of candidate_k under pi, not a hard
selection and not an input to the loss.

`hdbscan_min_samples` remains independent and fixed; it affects cluster and
support discovery, not candidate core distances. The legacy scalar `mrd_k`
remains relevant **only** when `point_selection: adaptive`, where the original
core/border support-selection heuristic uses it. Thus the new MRD_j is not a
single global metric independent of the cluster.

`support_distance: mrd` (default) uses the learned cluster-specific mixture of
core distances. `support_distance: euclidean` uses the same HDBSCAN clusters,
support points and support weights, but averages ordinary Euclidean distances
to those points in the clustering loss. The learned `candidate_k` mixture is
inactive in this mode: no scale logits are created or updated, and
`density_scale_info.active` is `false` with empty `pi` and `k_eff` lists.
`mrd_k` can still affect adaptive support selection. This is **not** KMeans
or distance to a centroid.

On HDBSCAN refresh, clusters are matched one-to-one by Jaccard overlap of all
member node IDs, with a minimum overlap of 0.25. Matching preserves the same
logit Parameter and optimizer state despite numeric relabeling. New clusters
start uniformly; vanished clusters are removed. If HDBSCAN produces fewer than
two valid clusters, the previous clustering state is retained, as in DensGNN.

Example using the root benchmark (set `models.deep_densgnn.enabled: true` in
`configs/config_newman_cora.yaml` first):

```bash
python scripts/main.py --config configs/config_newman_cora.yaml --model deep_densgnn
```

For a single Cora run with a two-dimensional latent space:

```bash
python scripts/main.py --config configs/deep_densgnn_cora_2d.yaml --model deep_densgnn
```

This config keeps the model parameters from `configs/final_configs/cora_final.yaml`
except for `embedding_channels: 2`, and saves results under `results/cora_2d/`.
With two embedding dimensions, the figure generator plots the original latent
coordinates directly, without standardization or t-SNE, and uses equal axis
scales. Higher-dimensional embeddings continue to use t-SNE. Each run saves
the plot to `deep_densgnn/figures/embedding_projections.png`.

The run's `deep_densgnn/metrics.json` contains
`algorithm.density_scale_info` with `cluster_labels`, `candidate_k`, `pi`, and
`k_eff`. `metadata.yaml` mirrors this. Set `log_density_scales: true` to also
save these diagnostics at each HDBSCAN refresh in `scale_history`. No
ground-truth labels are used during training; optional epoch NMI is calculated
after training.

Limits: the final HDBSCAN refresh can create a new cluster after the last
optimizer step; its pi is uniform until trained. Candidate k above N-1 is
clipped to N-1, so choose scales appropriate to graph size. Like the original,
the method materializes an N×N distance matrix and may be memory-intensive on
large graphs.
