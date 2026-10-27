import logging
import networkx as nx
from torch_geometric.utils import to_networkx


def select_centroids(data=None, Z=None, n_clusters=0):
    logging.info("Using Fast Greedy to find the centroids...")
    G = nx.Graph(to_networkx(data, node_attrs=["x"]))

    # using Fast Greedy
    communities = nx.community.greedy_modularity_communities(
        G, cutoff=n_clusters, best_n=n_clusters
    )

    # For each community, find the centroid
    centroids = __get_clusters_centroids(G, communities)

    clusters_centroids = []

    # Get Z values for each centroid.
    for c in centroids:
        clusters_centroids.append(Z[c].tolist())

    return clusters_centroids


def __get_clusters_centroids(G, communities):
    """
    Return the node that represents the centroid for each cluster.
    The centroid is chosen based on the centrality degrees of the node.
    """

    # TODO: Recheck this method.
    centrality_degrees = nx.degree_centrality(G)
    best_node_centrality = 0
    best_nodes = [0] * len(communities)
    comm_idx = 0

    for community in communities:
        for node in community:
            if centrality_degrees[node] > best_node_centrality:
                best_node_centrality = centrality_degrees[node]
                best_nodes[comm_idx] = node
        comm_idx += 1
        best_node_centrality = 0

    return best_nodes
