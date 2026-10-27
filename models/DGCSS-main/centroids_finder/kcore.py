import logging
import heapq
import networkx as nx
from torch_geometric.utils import to_networkx


def select_centroids(data=None, Z=None, n_clusters=0):
    logging.info("Using K-Core to find the centroids...")

    G = nx.Graph(to_networkx(data, node_attrs=["x"]))

    # This will return the k-core for each node.
    # {node_id: k-core, node_id: k-core, ...}
    core_numbers = nx.core_number(G)

    # Get the N nodes with the highest k-core score.
    highest_k_cores = heapq.nlargest(
        n_clusters, core_numbers.keys(), key=lambda k: core_numbers[k]
    )

    centroids = highest_k_cores

    clusters_centroids = []

    # Get Z values for each centroid.
    for c in centroids:
        clusters_centroids.append(Z[c].tolist())

    return clusters_centroids
