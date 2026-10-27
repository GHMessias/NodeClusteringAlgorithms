# VGAE + KMeans

This baseline trains a two-layer variational GCN graph autoencoder with the
PyTorch Geometric inner-product decoder. Its objective is edge reconstruction
plus the node-normalized KL divergence weighted by `kl_weight`. KMeans is then
applied to the posterior-mean embeddings emitted in evaluation mode.

Ground-truth labels do not participate in training and are used only to infer
`K` when no explicit cluster count is provided.
