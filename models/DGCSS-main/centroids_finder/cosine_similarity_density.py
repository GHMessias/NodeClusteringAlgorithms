import numpy as np
import torch
import logging
from sklearn.metrics.pairwise import cosine_similarity

def select_centroids(data, Z, n_clusters, threshold=0.0):
    """
    Selects cluster centroids based on cosine similarity between node embeddings,
    considering all node pairs regardless of the original graph structure.
    
    This function computes the full cosine similarity matrix for all node embeddings
    in Z. It then computes a "density" score for each node as the sum of cosine similarities 
    to every other node. Nodes with higher density (i.e., high overall similarity to all 
    other nodes) are selected as centroids.
    
    Args:
        data: A graph data object. (Not used directly here but kept for interface consistency.)
        Z (torch.Tensor or np.ndarray): Node embeddings of shape (num_nodes, embedding_dim).
        n_clusters (int): The number of centroids to select.
        threshold (float): Optional threshold to ignore very low similarity values 
                           (default is 0.0, i.e. no thresholding).
    
    Returns:
        list: A list of centroids, where each centroid is represented as a list of floats.
    """
    logging.info("Using Cosine Similarity Density (all connections) to find centroids...")

    # Ensure Z is a numpy array.
    if isinstance(Z, torch.Tensor):
        Z_np = Z.detach().cpu().numpy()
    else:
        Z_np = np.array(Z)

    # Compute the full cosine similarity matrix.
    sim_matrix = cosine_similarity(Z_np)
    # Clip similarities to avoid numerical issues (ensure values are in [-1,1]).
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
    
    # Optionally, zero out very small similarity values.
    sim_matrix[sim_matrix < threshold] = 0.0

    # Compute a "density" for each node: sum of cosine similarities for all node pairs.
    weighted_degree = np.sum(sim_matrix, axis=1)

    # Sort nodes by weighted degree in descending order.
    sorted_indices = np.argsort(-weighted_degree)

    # Select the top n_clusters nodes as centroids.
    centroids = [Z_np[idx].tolist() for idx in sorted_indices[:n_clusters]]
    
    return centroids