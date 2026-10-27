import logging
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx


def calculate(data, labels):
    """
    Calculate the modularity for the Graph clustering.
    Args:
        data (dataset): Data structure to build the Graph in Networkx format.
        labels (list): List with the labels for each node. 
    """
    logging.debug("Calculating Modularity...")
    G = nx.Graph(to_networkx(data))
    
    unique_labels = set(labels)
    
    nodes_sets = []
    
    # Transform the list of classes in a dict format.
    for label in unique_labels:
        nodes_sets.append(set(np.where(labels == label)[0]))

    mod = nx.community.modularity(G, nodes_sets)

    return mod
    
