import logging
import heapq
import networkx as nx
from torch_geometric.utils import to_networkx


def select_centroids(data=None, Z=None, n_clusters=0):
    logging.info("Using PageRank to find the centroids...")
    G = nx.Graph(to_networkx(data, node_attrs=["x"]))

    # Using PageRank to get centroids
    page_rank_nodes = nx.pagerank(G, alpha=0.9)
    biggest = heapq.nlargest(n_clusters, page_rank_nodes.items(), key=lambda i: i[1])
    centroids = [i[0] for i in biggest]

    clusters_centroids = []

    # Get Z values for each centroid.
    for c in centroids:
        clusters_centroids.append(Z[c].tolist())

    return clusters_centroids
