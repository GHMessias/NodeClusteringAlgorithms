import matplotlib.pyplot as plt
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


CENTROIDS_PLOT_FILENAME = "plots/centroids_plot.png"
CLUSTERING_FILENAME = "plots/clustering_plot.png"


def plot_centroids(all_nodes, centroids, filename=None):
    """
    Apply PCA (2) to all nodes and centroids and plot.
    Args:
        (list of list): all_nodes.
        (list of list): centroids (also nodes).
    """
    pca = PCA(n_components=2)
    all_nodes_pca = pca.fit_transform(all_nodes.cpu().detach().numpy())

    pca = PCA(n_components=2)
    centroids_pca = pca.fit_transform(centroids.detach().cpu().numpy())

    all_nodes_x = [row[0] for row in all_nodes_pca]
    all_nodes_y = [row[1] for row in all_nodes_pca]

    centroids_x = [row[0] for row in centroids_pca]
    centroids_y = [row[1] for row in centroids_pca]

    _plot(all_nodes_x, all_nodes_y, centroids_x, centroids_y, filename)

    return None


def _plot(all_nodes_x, all_nodes_y, centroids_x, centroids_y, filename=None):
    logging.info("Ploting centroids... saving into file.")

    # Add all nodes to the plot
    plt.scatter(all_nodes_x, all_nodes_y)

    # Add centroids nodes to the plot
    plt.scatter(centroids_x, centroids_y, c=["#FF0000"])

    if filename:
        plt.savefig(filename)
    else:
        plt.savefig(CENTROIDS_PLOT_FILENAME)


def plot_clustering(all_nodes_x, all_nodes_y, filename=CLUSTERING_FILENAME):
    logging.debug("Finding t-SNE 2 dimension data...")

    X_embedded = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=40
    ).fit_transform(all_nodes_x)

    logging.debug("Ploting clustering result and saving into file.")

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=all_nodes_y)

    plt.savefig(filename)

    plt.clf()
