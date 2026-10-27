# KMeans on node attributes

This baseline applies scikit-learn KMeans directly to `data.x`. It does not use
the graph topology and does not train a neural encoder. The saved
`embeddings.npy` is the original attribute matrix, so the benchmark's internal
metrics are evaluated in exactly the same space used by KMeans.

When `n_clusters` is omitted, the adapter infers the number of classes from
`data.y`. When executing this adapter directly on a dataset without single-label
ground truth, configure an explicit value. The central runner instead controls
K through its common `cluster_count` policy.

Run it through the benchmark with:

```bash
python scripts/main.py --config configs/global_config.yaml --model kmeans_attributes
```
