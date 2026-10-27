# DMoN benchmark adapter

This directory contains the repository-local node-clustering entry point built
with PyTorch Geometric's `DMoNPooling` operator. It applies a two-layer GCN
encoder to the configured node attributes and trains the unsupervised spectral
and cluster objectives returned by the pooling layer. Both GCN layers use the
width configured by `hidden_channels`; their count is configured by
`num_layers` (two by default in the benchmark configuration).

`DMoNPooling` consumes a dense adjacency matrix. Memory use therefore grows
quadratically with the node count, which should be considered before enabling
this entry for substantially larger graphs.

The benchmark normally invokes this model through `scripts/main.py`. It can
also be executed directly:

```bash
python models/DMoN/main.py \
  --data_pt datasets/Cora/processed/data.pt \
  --output_dir /tmp/dmon-output \
  --device cpu \
  --n_clusters 7
```

References:

- https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.nn.dense.DMoNPooling.html
- https://www.jmlr.org/papers/volume24/20-998/20-998.pdf
