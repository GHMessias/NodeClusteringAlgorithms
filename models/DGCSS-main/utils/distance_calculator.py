import logging
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity


def graph_attr_distances(X, consider_edges=False, mechanism="manhattan"):
    """
    Calculate the manhattan distance from all N to N nodes in a graph.

    Args:
        X ([[]]): The nodes attributes values.

        consider_edges (bool): If True, distance between nodes that doesn't
        have edge between then is infinite. If False, ignore edges in the
        and calculate based on the attributes.

        mechanism (str): the mechanism to be used to calculate the distances.

    Return:

        Return a matrix N to N, where N is the number of nodes, containing
        all the distances between nodes.

    """
    number_of_nodes = len(X)
    distances = [([0] * number_of_nodes) for i in range(number_of_nodes)]

    logging.info("Calculating distances between nodes.")
    logging.info("Number of nodes: " + str(number_of_nodes))
    logging.info("Distance mechanism: " + mechanism)

    """
    According to the Networkx docs:
    'This algorithm is not guaranteed to be correct if edge weights
    are floating point numbers. As a workaround you can use integer
    numbers by multiplying the relevant edge attributes by a convenient
    constant factor (eg 100) and converting to integers.'
    """

    if mechanism == "manhattan":
        distances = manhattan_distances(X, X)

    elif mechanism == "cosine":
        distances = cosine_similarity(X)

    else:
        logging.error("Distance mechanism not known. Aborting...")
        return None

    return distances


def define_weights(G, distances, weight_name, multiplier="direct", scale=1000):
    """
    Define the weights in the Graph based on the distance between nodes.
    The weight is defined by 1/<distance>.
    Args:
        G: Networkx Graph.
        distances ([[]]): Matrix NxN with all the distances between nodes.
        weight_name (str): Name of the field that will be stored in the Graph.
        multiplier (str): "inverse" or "direct". Determine the correlation
          between the distance and the weights.
        scale (int): multiplier for weights.
    Return:
        Graph attributes as a dict. {(tuple, tuple): value}
    """
    logging.info(
        str("Defining graph weights based on distances. Correlation: " + multiplier)
    )

    g_attrs = {}

    number_of_nodes = len(distances)

    # TODO: Need to review this.
    if multiplier == "inverse":
        for i in range(number_of_nodes):
            for j in range(number_of_nodes):
            
                # If the edge doesn't exists, we don´t need to calculate distance.
                if not G.has_edge(i, j):
                    pass

                distance = distances[i][j]

                if distance <= 0.001:
                    weight = 10000

                else:
                    weight = (1 / distance)

                g_attrs[(i, j)] = {weight_name: weight}

    elif multiplier == "direct":
        for i in range(number_of_nodes):
            for j in range(number_of_nodes):
                weight = int(pow(distances[i][j], 4) * scale * 100)
                g_attrs[(i, j)] = {weight_name: weight}

    else:
        logging.error("Invalid value for MULTIPLIER. Must be 'inverse' or 'direct'.")
        return {}

    return g_attrs
