import logging
from sklearn.cluster import KMeans


def select_centroids(
    data=None,
    Z=None,
    n_clusters=0,
    random_state=0,
    n_init="auto",
):
    """
    Runs KMeans clustering to find the centroids.
    """

    if Z is None:
        logging.error("Missing Z. Aborting...")
        return None

    logging.info("Using K-Means to find centroids.")
    X = Z.cpu().detach().numpy()

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    ).fit(X)
    return kmeans.cluster_centers_
