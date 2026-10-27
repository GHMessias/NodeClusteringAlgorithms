# GAE + KMeans

This benchmark baseline trains a two-layer GCN graph autoencoder using the
standard inner-product edge-reconstruction objective from PyTorch Geometric.
The final node embeddings are clustered with scikit-learn KMeans.

The adapter saves `embeddings.npy`, a timestamped `*_labels.npy`, KMeans cluster
centers, the epoch loss history, and a JSON run summary in the configured output
directory. Ground-truth labels are used only to infer `K` when no explicit
cluster count is supplied; they are not used during representation learning.
