# ARGA + KMeans

This baseline trains an adversarially regularized graph autoencoder with a
two-layer GCN encoder, the PyTorch Geometric inner-product decoder, and an MLP
discriminator. The discriminator matches encoded nodes to a standard Gaussian
prior; the final graph embeddings are clustered with KMeans.

Encoder and discriminator losses are saved separately in
`training_history.csv`. Ground-truth labels are never used by either optimizer
and only provide `K` when no explicit cluster count is configured.
