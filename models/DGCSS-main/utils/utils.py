import logging

import torch
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment


logger = logging.getLogger(__name__)


def edges_to_edgeindex(edges):
    '''
    Transforma uma lista de edge index do formato
    [(x1,y1), (x2,y2)] no formato [[x1,x2], [y1,y2]]
    '''
    src = [x[0] for x in edges]
    tgt = [x[1] for x in edges]
    return torch.tensor([src, tgt])


def remove_min_weight_edges(graph):
    '''
    Remove edges with less weights.
    '''

    # Find the min weight from edges
    min_weight = min(graph.edges(data=True),
                     key=lambda x: x[2]['weight'])[-1]['weight']

    # Map all edges with this weight
    edges_to_remove = [
        (u, v) for u, v, data in graph.edges(data=True)
        if data['weight'] == min_weight]

    # Remove it
    graph.remove_edges_from(edges_to_remove)

    return graph


# Transforma a tupla em matriz de adjacência (somente para visualização)
def tuple_to_adj(att_tuple, G):
    # att_tuple[1] -> lista de pesos de cada aresta no grafo
    # att_tuple[0][0] -> lista de source de cada aresta
    # att_tuple[0][1] -> lista de target de cada aresta
    adj = torch.zeros((len(G.nodes()), len(G.nodes())))
    for i in range(len(att_tuple[1])):
        adj[att_tuple[0][0][i], att_tuple[0][1][i]] = att_tuple[1][i]

    return adj, att_tuple[1]


def remove_edges(G, x):
    # Remove weights with small weights, based on the Attention values.
    logger.info("Removing edges with small Attention values...")

    num_rem = 0
    while check_size_of_groups(G, x) is False:
        G = remove_min_weight_edges(G)
        num_rem += 1

    logger.info("Removed %s edges.", num_rem)

    return G


def check_size_of_groups(G, x):
    '''
    Consider a Graph division as valid when we don't have any
    cluster (group) bigger than 1/x size of the group.
    '''
    valid_size = True

    fraction_size_graph = round(len(G.nodes)/x)

    all_groups = list(nx.connected_components(G))

    # print("==> Longest list: ", len(max(all_groups, key=len)),
    #       " - num clusters: ", len(all_groups))

    for group in all_groups:
        if len(group) > fraction_size_graph:
            valid_size = False

    return valid_size

def clustering_accuracy(y_true, y_pred):
    """
    Compute clustering accuracy using the Hungarian algorithm.
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted cluster labels.
    
    Returns:
        float: Accuracy, in [0, 1].
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Ensure labels start at 0
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    # Build contingency matrix (rows: predicted, columns: true)
    contingency_matrix = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_true)):
        contingency_matrix[y_pred[i], y_true[i]] += 1

    # Solve the linear assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Sum the matched entries for the best accuracy
    total = contingency_matrix[row_ind, col_ind].sum()
    return total / len(y_true)
