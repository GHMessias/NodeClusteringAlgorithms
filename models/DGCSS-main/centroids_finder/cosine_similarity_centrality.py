import numpy as np
import torch
import logging
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def select_centroids(data, Z, n_clusters):
    """
    Selects cluster centroids based on cosine similarity between node embeddings,
    but strictly respects the original graph structure by only considering edges
    that exist in the original dataset (as defined by data.edge_index).

    This function computes the cosine similarity between all node embeddings in Z,
    then masks out similarities for node pairs that are not connected in the original
    graph. The resulting masked similarity matrix is used to build a weighted graph,
    where the distance is defined as (1 - similarity). Closeness centrality is computed
    on this weighted graph, and nodes with the highest centrality are chosen as centroids.

    Args:
        data: A graph data object which must contain an attribute 'edge_index' (shape [2, num_edges]).
        Z (torch.Tensor or np.ndarray): Node embeddings of shape (num_nodes, embedding_dim).
        n_clusters (int): The number of centroids to select.

    Returns:
        list: A list of centroids, where each centroid is represented as a list of floats.
    """
    logging.info("Using Cosine Similarity with Original Graph Structure to find centroids...")

    # Ensure that Z is a numpy array.
    if isinstance(Z, torch.Tensor):
        Z_np = Z.detach().cpu().numpy()
    else:
        Z_np = np.array(Z)

    num_nodes = Z_np.shape[0]

    # Compute the full cosine similarity matrix.
    sim_matrix = cosine_similarity(Z_np)
    # Clip similarities to avoid numerical issues (values > 1 can cause negative distances).
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

    # Create a mask based on the original graph structure.
    # We expect data.edge_index to be of shape [2, num_edges]
    if hasattr(data, 'edge_index'):
        edge_index = data.edge_index
        if isinstance(edge_index, torch.Tensor):
            edge_index_np = edge_index.detach().cpu().numpy()
        else:
            edge_index_np = np.array(edge_index)
    else:
        raise AttributeError("The provided data object does not have an 'edge_index' attribute.")

    # Initialize a mask with zeros.
    mask = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # For every edge (u,v) in the original graph, mark both [u,v] and [v,u] as valid.
    u = edge_index_np[0]
    v = edge_index_np[1]
    mask[u, v] = 1.0
    mask[v, u] = 1.0

    # Optionally, you can set the diagonal to 1 (each node is connected to itself).
    np.fill_diagonal(mask, 1.0)

    # Apply the mask: only similarities for pairs connected in the original graph are kept.
    sim_matrix_filtered = sim_matrix * mask

    # Create a weighted graph from the filtered similarity matrix.
    G = nx.from_numpy_array(sim_matrix_filtered)

    # Convert similarity into a "distance" for centrality computation:
    # Define distance = 1 - similarity. With the mask, unconnected pairs have similarity 0 (distance 1).
    for u, v, attr in G.edges(data=True):
        similarity = attr.get('weight', 1.0)
        attr['distance'] = 1.0 - similarity

    # Compute closeness centrality using the custom 'distance' attribute.
    centrality = nx.closeness_centrality(G, distance='distance')

    # Sort nodes by centrality in descending order (highest centrality first).
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    # Select the top n_clusters nodes as centroids.
    centroids = [Z_np[node_idx].tolist() for node_idx, _ in sorted_nodes[:n_clusters]]
    
    return centroids