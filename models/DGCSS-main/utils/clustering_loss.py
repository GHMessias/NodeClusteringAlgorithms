import torch
import torch.nn.functional as F
import logging


def calculate_q(cluster_centroids, Z, alpha=1.0):
    """
    Calculate the soft assignment matrix Q using Student’s t-distribution.
    
    For each data point z_i and cluster center μ_j, the soft assignment is:
    
        q_ij = (1 + ||z_i - μ_j||² / alpha)^(-(alpha+1)/2)
               --------------------------------------------------
               sum_k (1 + ||z_i - μ_k||² / alpha)^(-(alpha+1)/2)
    
    Args:
        cluster_centroids (torch.Tensor): Tensor of cluster centers with shape 
                                          (n_clusters, embedding_dim). This tensor 
                                          should have requires_grad=True if you plan to
                                          update it via gradients.
        Z (torch.Tensor): Data embeddings with shape (n_samples, embedding_dim).
        alpha (float): Parameter for the Student’s t-distribution.
    
    Returns:
        torch.Tensor: Soft assignment matrix Q with shape (n_samples, n_clusters).
    """
    # Compute pairwise squared Euclidean distances.
    # Expand dimensions so that:
    #   Z becomes (n_samples, 1, embedding_dim)
    #   cluster_centroids becomes (1, n_clusters, embedding_dim)
    diff = Z.unsqueeze(1) - cluster_centroids.unsqueeze(0)  # (n_samples, n_clusters, embedding_dim)
    distances = torch.sum(diff ** 2, dim=2)  # (n_samples, n_clusters)
    
    # Compute the numerator of Q using the Student’s t-distribution kernel.
    numerator = (1.0 + distances / alpha) ** (-(alpha + 1) / 2)
    
    # Normalize each row so that the assignments sum to 1.
    Q = numerator / torch.sum(numerator, dim=1, keepdim=True)
    return Q


def calculate_p(Q, eps=1e-8):
    """
    Calculate the target distribution P from Q.
    
    The target distribution is computed to "sharpen" Q. One common formulation is:
    
        p_ij = (q_ij² / f_j) / sum_k (q_ik² / f_k)
    
    where f_j = sum_i q_ij.
    
    Args:
        Q (torch.Tensor): Soft assignment matrix with shape (n_samples, n_clusters).
        eps (float): A small value to avoid division by zero.
    
    Returns:
        torch.Tensor: Target distribution P with shape (n_samples, n_clusters).
    """
    logging.info("Calculating P...")
    # Compute the frequency for each cluster (f_j).
    f = torch.sum(Q, dim=0, keepdim=True)  # shape: (1, n_clusters)
    # Compute weighted Q (square Q and normalize by f_j)
    weight = Q ** 2 / (f + eps)
    # Normalize each row to sum to 1.
    P = weight / torch.sum(weight, dim=1, keepdim=True)
    return P


def kl_div_loss(Q, P):
    """
    Calculate the clustering loss using PyTorch's KLDivLoss function.
    
    This method wraps the KL divergence calculation and returns the loss as well as Q and P,
    in case you want to inspect them.
    
    Args:
        Q (torch.Tensor): Soft assignment matrix with shape (n_samples, n_clusters).
        P (torch.Tensor): Target distribution with shape (n_samples, n_clusters).
    
    Returns:
        tuple: (loss, Q, P) where loss is a scalar tensor.
    """
    loss = F.kl_div(Q.log(), P.detach(), reduction="batchmean")
    return loss, Q, P
