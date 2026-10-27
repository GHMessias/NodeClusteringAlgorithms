import heapq
import logging
import networkx as nx
from torch_geometric.utils import to_networkx
from utils import distance_calculator


def select_centroids(data=None, Z=None, n_clusters=0):
    """
    Select centroids based on closeness centrality computed on the graph built from the original dataset.

    This function converts the provided torch_geometric data into a NetworkX graph (using the 'x' attribute for nodes),
    computes the closeness centrality for all nodes, and selects the top nodes with the highest centrality scores.
    It then returns the corresponding entries from Z for these nodes.

    Args:
        data (torch_geometric.data.Data): Graph data to be converted into a NetworkX graph.
        Z (list, Tensor, or array-like): The data corresponding to each node, from which centroids will be returned.
        n_clusters (int): The number of centroids (i.e., top nodes) to select.

    Returns:
        List: A list of centroids from Z corresponding to the nodes with the highest closeness centrality.
    """
    if data is None or Z is None:
        raise ValueError("Both 'data' and 'Z' must be provided.")

    logging.info("Computing closeness centrality on the original dataset to determine centroids.")

    # Convert the torch_geometric Data object to a NetworkX graph using the node attribute 'x'
    G = to_networkx(data, node_attrs=["x"])
    
    # Compute closeness centrality for each node in the graph.
    centrality = nx.closeness_centrality(G)

    num_nodes = len(centrality)
    if n_clusters > num_nodes:
        logging.warning("n_clusters (%d) is greater than the number of nodes (%d). Using all available nodes.",
                        n_clusters, num_nodes)
        n_clusters = num_nodes

    # Sort nodes by centrality in descending order and select the top n_clusters nodes.
    sorted_nodes = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
    top_node_indices = [node for node, _ in sorted_nodes[:n_clusters]]

    # Map the selected node indices to their corresponding Z values.
    centroids = []
    for idx in top_node_indices:
        node_value = Z[idx]
        # Convert to a list if the node value is a tensor or array-like with a tolist() method.
        if hasattr(node_value, "tolist"):
            node_value = node_value.tolist()
        centroids.append(node_value)

    return centroids

