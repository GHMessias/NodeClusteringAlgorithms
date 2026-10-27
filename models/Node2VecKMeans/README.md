# Node2Vec + KMeans

This structural baseline learns Node2Vec embeddings from `data.edge_index` with
`torch_geometric.nn.Node2Vec` and then applies scikit-learn KMeans. Node
attributes are intentionally ignored.

PyG uses the installed `torch_cluster` extension for its second-order random
walks. The walk bias (`p` and `q`), skip-gram negative sampling, and sparse
embedding optimization are handled through PyG's Node2Vec interface. Training
supports CPU and CUDA.

When `n_clusters` is omitted, the adapter infers the number of classes from
`data.y`. When executing this adapter directly on a dataset without single-label
ground truth, configure an explicit value. The central runner instead controls
K through its common `cluster_count` policy.

Run it through the benchmark with:

```bash
python scripts/main.py --config configs/global_config.yaml --model node2vec_kmeans
```
