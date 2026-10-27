import heapq
import logging
import networkx as nx
from torch_geometric.utils import to_networkx
from utils import distance_calculator


def select_centroids(data=None, Z=None, n_clusters=0):
    logging.info("Using Eigen Vector Centrality to find the centroids...")
    G = nx.Graph(to_networkx(data))
  
    # Calculating diatances via COSINE SIMILARITY. 1 = equal, 0 = very different.
    distances = distance_calculator.graph_attr_distances(Z.detach().numpy(), mechanism="cosine")

    g_attrs = distance_calculator.define_weights(
        G=G, distances=distances, weight_name="distancia", multiplier="direct"
    )

    nx.set_edge_attributes(G, g_attrs)
    
    # Get a sample of weights, just to check.
    edges_weights = list(nx.get_edge_attributes(G, "distancia").items())[:5]
    msg = "Edge weights sample: " + str(edges_weights)
    logging.info(msg)
   
    # In this measure the weight is interpreted as the connection strength.
    ev_nodes = nx.eigenvector_centrality(G, max_iter=1000, weight="distancia")

    biggest = heapq.nlargest(n_clusters, ev_nodes.items(), key=lambda i: i[1])
    centroids = [i[0] for i in biggest]

    clusters_centroids = []

    # Get Z values for each centroid.
    for c in centroids:
        clusters_centroids.append(Z[c].tolist())

    return clusters_centroids
