import argparse
import logging
import sys

from runners import gae_runner
from utils.graph_creator import get_dataset
from utils.b_matrix import BMatrix

# Define dataset names for each group
PLANETOID_DATASETS = {"Cora", "Citeseer", "Pubmed"}
TWITCH_DATASETS = {"Twitch"}
COAUTHOR_DATASETS = {"CS", "Physics"}
ACTOR_DATASET = {"Actor"}
AMAZON_DATASETS = {"Computers", "Photo"}
B_MATRIX_DEGREE = 3


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run GAE experiments.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "-fa", "--find_centroids_alg",
        type=str,
        default="KMeans",
        help=(
            "Method to find centroids. Options: KMeans, FastGreedy, WFastGreedy, Random, "
            "BC (Betweenness Centrality), WBC (Weighted Betweenness Centrality), "
            "PageRank, KCore, EigenV (Eigen Vector), CC (Closeness Centrality)"
        )
    )
    parser.add_argument("-ds", "--dataset_name", type=str, default="Cora", help="Dataset name to use.")
    parser.add_argument("-log", "--loss_log_file", type=str, default="loss_log.csv", help="CSV file for loss logs.")
    parser.add_argument("-metrics", "--metrics_log_file", type=str, default="metrics_log.csv", help="CSV file for metrics logs.")
    parser.add_argument("-cl", "--c_loss_gama", type=int, default=20, help="Multiplier for Clustering Loss.")
    parser.add_argument("-pi", "--p_interval", type=int, default=10, help="Interval for calculating P.")
    parser.add_argument("-hl", "--hidden_layer", type=int, default=64, help="Hidden layer size.")
    parser.add_argument("-ol", "--output_layer", type=int, default=16, help="Output layer size.")
    parser.add_argument("--centroids_plot_file", type=str, default="plots/centroids_plot.png", help="PNG file for centroids plot.")
    parser.add_argument("--clustering_plot_file", type=str, default="plots/clustering_plot.png", help="PNG file for clustering plot.")
    return parser.parse_args()


def configure_logging(debug: bool):
    """Configure the logging settings."""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=log_level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def determine_dataset_type(dataset_name: str) -> str:
    """Return the dataset type based on the dataset name."""
    if dataset_name in PLANETOID_DATASETS:
        return "Planetoid"
    if dataset_name in TWITCH_DATASETS:
        return "Twitch"
    if dataset_name in COAUTHOR_DATASETS:
        return "Coauthor"
    if dataset_name in ACTOR_DATASET:
        return "Actor"
    if dataset_name in AMAZON_DATASETS:
        return "Amazon"
    return None


def main():
    args = parse_args()
    configure_logging(args.debug)

    logging.info("Chosen dataset: %s", args.dataset_name)
    logging.info("Running for %s epochs", args.epochs)

    ds_type = determine_dataset_type(args.dataset_name)
    if ds_type is None:
        logging.error("Dataset '%s' not recognized. Aborting...", args.dataset_name)
        sys.exit(1)

    # Load the dataset and log basic info
    dataset = get_dataset(name=args.dataset_name, ds_type=ds_type)
    data = dataset[0]
    num_classes = dataset.num_classes
    num_nodes = len(data.x)
    logging.info("Number of nodes: %d", num_nodes)
    logging.info("Number of classes: %d", num_classes)

    # Create and process the BMatrix
    b_matrix = BMatrix(data)
    logging.debug("B Matrix: %s", str(b_matrix))
    logging.info("Calculating B Matrix...")
    b_matrix.calc_t_order_neighbors(data, t=B_MATRIX_DEGREE)
    b_matrix.create_edge_index()
    logging.info("Done.")

    # Instantiate the GAE runner with provided parameters
    runner = gae_runner.GaeRunner(
        epochs=args.epochs,
        data=data,
        b_edge_index=b_matrix.edge_index,
        n_clusters=num_classes,
        find_centroids_alg=args.find_centroids_alg,
        c_loss_gama=args.c_loss_gama,
        p_interval=args.p_interval,
        centroids_plot_file=args.centroids_plot_file,
        clustering_plot_file=args.clustering_plot_file,
        loss_log_file=args.loss_log_file,
        metrics_log_file=args.metrics_log_file,
        hidden_layer=args.hidden_layer,
        output_layer=args.output_layer
    )

    # Run the training and retrieve any additional outputs (like attention values)
    data, att_tuple = runner.run_training()
    logging.debug("Attention values: %s", str(att_tuple))


if __name__ == "__main__":
    main()
