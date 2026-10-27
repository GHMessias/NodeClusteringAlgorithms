"""
Deprecated file. This approach is not working as expected.
We may try again in the future.
"""

import networkx as nx
import itertools
import logging
import copy
import numpy as np


def run_gn(G, number_of_groups):
    """
    Execute the Girvan Newman algorithm and find the best communities
    set for the number of communities specified.
    Args:
        (NetworkX Graph format): NetworkX Graph
        (int) number_of_groups
    Return:
        (tuple): communities definition
    """
    logging.info("Running Girvan Newman algorithm...")

    k = number_of_groups

    comp = nx.community.girvan_newman(G)
    limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    communities_final = None

    for communities in limited:
        communities_final = tuple(sorted(c) for c in communities)

    return communities_final


def get_modularity(G, communities):
    return nx.community.modularity(G, communities)


def calculate_Q_distribution_graph(G, communities, mod_score):
    """
    For each node present in the Gragh G, calculate the modularity score
    variations.
    """
    logging.info("Calculating modularity variation for Graph...")
    all_mod_variations = []
    for node in G.nodes():
        all_mod_variations.append(calculate_Q_distribution_single_node(
            G, node, communities, mod_score))

    return all_mod_variations


def calculate_Q_distribution_single_node(G, node, communities, mod_score):
    """
    Considering a node X, a graph G and the communities
    distribution 'communities', calculate the modularity Q
    for every community that X could be part of, so Qx
    would be a distribution of modularities for X being part
    of all communities.
    Args:
        (NetworkX Graph): G as the full graph.
        (int): the index of the node.
        (communities): tuple of lists being the communities.
        (float): Original modularity score. (the best one)
    Return:
        (list): Qx, the modularities distributions.
    """
    # Testing...

    node_comm = 0
    node_modularity_scores = [0] * len(communities)

    # Create a new copy so we don't change the original.
    comm_copy = copy.deepcopy(communities)

    for c_idx in range(len(comm_copy)):
        if node in comm_copy[c_idx]:
            node_comm = c_idx

    node_modularity_scores[node_comm] = mod_score

    current_node_community = node_comm

    for c_idx in range(len(comm_copy)):
        if c_idx == node_comm:
            continue

        # Change node community
        comm_copy[current_node_community].remove(node)
        comm_copy[c_idx].append(node)

        # Save where the node is. Which community.
        current_node_community = c_idx

        # Calculate modularity score for this new set of communities.
        node_modularity_scores[c_idx] = get_modularity(G, comm_copy)

    norm_scores = node_modularity_scores / np.sum(node_modularity_scores)

    return norm_scores
