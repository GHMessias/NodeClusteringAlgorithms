from centroids_finder import (
    random_seeds,
    fastgreedy,
    kcore,
    weighted_modularity,
    pagerank,
    kmeans,
    betweenness_centrality,
    weighted_betweenness_centrality,
    eigenvector_centrality,
    closeness_centrality,
    cosine_similarity_centrality,
    cosine_similarity_density
)


# Function to calculate centroids will be called according to
# this map. All the functions must have the same method name and arguments.
# Example: select_centroids(data=None, Z=None, n_clusters=0)

map = {
    "Random": random_seeds,
    "BC": betweenness_centrality,
    "WBC": weighted_betweenness_centrality,
    "PageRank": pagerank,
    "KMeans": kmeans,
    "FastGreedy": fastgreedy,
    "KCore": kcore,
    "EigenV": eigenvector_centrality,
    "CC": closeness_centrality,
    "CSC": cosine_similarity_centrality,
    "CSD": cosine_similarity_density
}
