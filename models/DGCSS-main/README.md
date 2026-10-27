# DGCSS (Deep Graph Clustering with Seed Selection)

## Overview

In this study, we propose **DGCSS**, a novel approach designed to push the boundaries of state-of-the-art performance in node clustering within complex networks. Traditional deep clustering methods often rely on external clustering algorithms to identify representative elements, but these methods can be hampered by the influence of initially formed groups and typically focus only on the content information of each example, neglecting the rich topological structure of the data.

DGCSS addresses these limitations with a three-module design:

- **Embedding Module:** Utilizes a graph attentional network to capture topological information from the data.
- **Seed Selection Module:** Detects representative nodes (seeds) in the graph, ensuring robust initial groupings.
- **Self-Supervised Module:** Leverages the detected representative nodes to guide the clustering process and refine results.

Our goal is to establish DGCSS as a state-of-the-art method for deep graph clustering by effectively integrating both content and topological information to achieve superior clustering performance.

![DGCSS Architecture](/img/dgcsd-v2.png)


## Features

- **Enhanced Clustering Accuracy:** Mitigates the limitations of external clustering algorithms by directly integrating seed detection.
- **Topological Awareness:** Incorporates graph attentional networks to capture essential structural information.
- **Self-Supervision:** Uses detected seeds to iteratively refine clustering results.
- **Modular Architecture:** Simplifies customization and extension of the algorithm for various applications.

## Centroid Finder Algorithms

Our project supports a variety of centroid selection algorithms that can be leveraged during training to initialize or update cluster centroids. You can choose the algorithm that best fits your data and use-case by specifying its corresponding key in the configuration. The available algorithms are:

#### Core:

- **Random**: Uses random seeding to select centroids.
- **BC (Betweenness Centrality)**: Selects centroids based on betweenness centrality.
- **CC (Closeness Centrality)**: Selects centroids by measuring the closeness centrality.

#### Extras:

- **WBC (Weighted Betweenness Centrality)**: Similar to BC but accounts for edge weights.
- **PageRank**: Applies the PageRank algorithm to determine influential nodes as centroids.
- **KMeans**: Utilizes the K-Means clustering algorithm for centroid initialization.
- **FastGreedy**: Employs the Fast Greedy algorithm for community detection.
- **KCore**: Uses K-Core decomposition to identify core nodes as centroids.
- **EigenV (Eigenvector Centrality)**: Chooses centroids based on eigenvector centrality.
- **CSC (Cosine Similarity Centrality)**: Uses cosine similarity to evaluate centrality.
- **CSD (Cosine Similarity Density)**: Determines centroids based on the density of cosine similarity.


## Installation

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/DGCSS.git
   cd DGCSS
   pip install -r requirements.txt
   python run.py --help


## Authors:

Carlos Pereira Lopes Filho

Guilherme Henrique Messias

Alan Demétrius Baria Valejo
