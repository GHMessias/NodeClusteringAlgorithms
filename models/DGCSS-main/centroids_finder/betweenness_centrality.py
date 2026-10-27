import heapq
import logging
import networkx as nx
from torch_geometric.utils import to_networkx


def select_centroids(data=None, Z=None, n_clusters=0):
    logging.info("Using Betweenness Centrality to find the centroids...")
    G = nx.Graph(to_networkx(data, node_attrs=["x"]))

    bc_nodes = nx.betweenness_centrality(G)
    biggest = heapq.nlargest(n_clusters, bc_nodes.items(), key=lambda i: i[1])
    centroids = [i[0] for i in biggest]

    clusters_centroids = []

    # Get Z values for each centroid.
    for c in centroids:
        clusters_centroids.append(Z[c].tolist())

    return clusters_centroids
