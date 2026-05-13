import math
import argparse
import os
import pickle as pkl
import time

from UCODEncoder import GCN
import utils
from Lossfunction import loss_modularity_trace

from tqdm.auto import tqdm
import networkx as nx
from networkx.generators.degree_seq import expected_degree_graph

from sklearn.cluster import KMeans
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy import sparse
from sklearn import metrics


def torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def get_data_attr(data, name):
    if isinstance(data, dict):
        return data.get(name)
    return getattr(data, name, None)


def unwrap_data_pt(obj):
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if get_data_attr(item, 'x') is not None and get_data_attr(item, 'edge_index') is not None:
                return item
    return obj


def as_numpy(value):
    if hasattr(value, 'is_sparse') and value.is_sparse:
        value = value.to_dense()
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def labels_to_vector(y):
    y = as_numpy(y)
    if y.ndim > 1:
        y = np.argmax(y, axis=1)
    return y.reshape(-1).astype(np.int64)


def adjacency_from_edge_index(edge_index, num_nodes):
    edge_index = as_numpy(edge_index)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    rows = edge_index[0].astype(np.int64)
    cols = edge_index[1].astype(np.int64)
    values = np.ones(rows.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def graph_from_adjacency(adj):
    if hasattr(nx, 'from_scipy_sparse_array'):
        return nx.from_scipy_sparse_array(adj)
    return nx.from_scipy_sparse_matrix(adj)


def load_data_pt(data_pt):
    data = unwrap_data_pt(torch_load(data_pt))
    x = get_data_attr(data, 'x')
    y = get_data_attr(data, 'y')
    edge_index = get_data_attr(data, 'edge_index')
    if x is None or y is None or edge_index is None:
        raise ValueError("data.pt must contain x, y and edge_index.")

    features = sparse.csr_matrix(as_numpy(x).astype(np.float32))
    labels = labels_to_vector(y)
    adj = adjacency_from_edge_index(edge_index, features.shape[0])
    network = graph_from_adjacency(adj)
    B = utils.get_B(network)
    label_mask = np.ones(labels.shape[0], dtype=bool)

    adj_metric = adj
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    m = len(network.edges)
    features, adj, B, true_labels = utils.convert_torch_npz(adj, features, B, labels)

    return features, adj, B, true_labels, label_mask, nb_nodes, ft_size, adj_metric, m


def datapreprocessing(data_type, path, dataset, exist_B):
    if data_type == 'kipf':
        adj, features, labels = utils.load_data(dataset)
        with open(path + "/dataset/ind." + str(dataset) + ".graph", 'rb') as f:
            Graph = pkl.load(f, encoding='latin1')
        network = nx.Graph(Graph)
        label_mask = np.ones_like(labels)
    elif data_type == "npz":
        adj, features, labels, label_mask = utils.load_npz_to_sparse_graph(path + '/dataset/' + dataset + '.npz')
        network = nx.from_scipy_sparse_matrix(adj)
        features = sparse.csr_matrix(features)
    adj_metric=adj
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    m = len(network.edges)

    if exist_B == 0:
        B=utils.get_B(network)
        np.save(path+'/dataset/Modularity-'+dataset, B)
    else:
        B = np.load(path+'/dataset/Modularity-'+dataset+'.npy')

    if data_type=="kipf":
        features, adj, B, true_labels = utils.convert_torch_kipf(adj, features, B, labels)
    elif data_type=="npz":
        features, adj, B, true_labels = utils.convert_torch_npz(adj, features, B, labels)

    return features, adj, B, true_labels, label_mask, nb_nodes, ft_size, adj_metric, m

def train(
    data_name,
    n_communities,
    preprocessed_data,
    hid_units,
    hid_dimension,
    device,
    epochs=500,
    num_experiments=1,
):
    sigmoidlogit = nn.Sigmoid()
    print(data_name)
    features, adj, B, true_labels, label_mask, nb_nodes, ft_size, adj_metric, m = preprocessed_data
    print(features.shape)
    nmi_list = np.zeros(num_experiments)

    kmeans_nmi_list = np.zeros(num_experiments)
    cond_list, kmeans_cond_list = np.zeros(num_experiments), np.zeros(num_experiments)
    modularity_list, kmeans_modularity_list = np.zeros(num_experiments), np.zeros(num_experiments)
    Fscore_list, kmeans_Fscore_list = np.zeros(num_experiments), np.zeros(num_experiments)

    final_kmeans_preds = None
    for i in tqdm(range(num_experiments)):
        model = GCN(ft_size, hid_units, nb_nodes,hid_dimension).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.1)
        features_device = features.to(device)
        B_device = B.to(device)
        adj_device = adj.to(device)

        # Training loop
        for epoch in range(epochs):
            model.train()
            optimiser.zero_grad()
            logits = model(features_device, adj_device)
            loss = loss_modularity_trace(logits, B_device, n_communities, hid_units, m,'non-overlap')
            #print('loss:',loss)
            loss.backward()
            optimiser.step()
            model.eval()
        model.eval()
        logits = model(features_device, adj_device)
        logits=sigmoidlogit(logits)
        cpu_logits = logits.view(nb_nodes, hid_units).detach().cpu().numpy()
        preds = torch.argmax(logits[0], dim=1).detach().cpu().numpy()
        nmi = normalized_mutual_info_score(true_labels, preds)
        print('nmi:'+str(nmi))
        # Fit kmeans to the output space
        kmeans_model = KMeans(
            init='k-means++',
            n_clusters=n_communities,
            random_state=0
        ).fit(cpu_logits)
        kmeans_preds = kmeans_model.labels_
        final_kmeans_preds = kmeans_preds

        # Calculate NMI
        nmi = normalized_mutual_info_score(true_labels, preds)
        kmeans_nmi = normalized_mutual_info_score(true_labels, kmeans_preds)
        kmeans_nmi_list[i] = kmeans_nmi
        nmi_list[i] = nmi

        cond_list[i] = utils.conductance(adj_metric, preds)
        kmeans_cond_list[i] = utils.conductance(adj_metric, kmeans_preds)
        modularity_list[i] = utils.modularity(adj_metric, preds)
        kmeans_modularity_list[i] = utils.modularity(adj_metric, kmeans_preds)
        p = utils.precision(true_labels, preds)
        r = utils.recall(true_labels, preds)
        Fscore_list[i] = 2 * (r * p) / (r + p)

        p = utils.precision(true_labels, kmeans_preds)
        r = utils.recall(true_labels, kmeans_preds)
        kmeans_Fscore_list[i] = 2 * (r * p) / (r + p)

    print('\n')
    print('Baseline scores:')
    print('NMI:', np.mean(nmi_list), np.std(nmi_list))
    print('Fscore:', np.mean(Fscore_list), np.std(Fscore_list))
    print('Conductance:', np.mean(cond_list), np.std(cond_list))
    print('Modularity:', np.mean(modularity_list), np.std(modularity_list))

    print('Kmeans scores:')
    print('NMI:', np.mean(kmeans_nmi_list), np.std(kmeans_nmi_list))
    print('Fscore:', np.mean(kmeans_Fscore_list), np.std(kmeans_Fscore_list))
    print('Conductance:', np.mean(kmeans_cond_list), np.std(kmeans_cond_list))
    print('Modularity:', np.mean(kmeans_modularity_list), np.std(kmeans_modularity_list))
    return final_kmeans_preds


def run_nonoverlapping(args):
    device = torch.device(args.device)
    preprocessed_data = load_data_pt(args.data_pt)
    true_labels = preprocessed_data[3]
    n_communities = args.kmeans_clusters
    if n_communities is None:
        n_communities = len(np.unique(true_labels))
    preds = train(
        data_name=os.path.basename(args.data_pt),
        n_communities=n_communities,
        preprocessed_data=preprocessed_data,
        hid_units=args.hid_units,
        hid_dimension=args.hid_dimension,
        device=device,
        epochs=args.epochs,
        num_experiments=args.num_experiments,
    )
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M")
        output_path = os.path.join(args.output_dir, "UCODE_{}_labels.npy".format(timestamp))
        np.save(output_path, preds.astype(np.int64))
        print("UCODE labels saved in: {}".format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UCODE non-overlapping community detection')
    parser.add_argument('--data_pt', type=str, required=True, help='Path to PyG data.pt with x, y and edge_index')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--kmeans_clusters', type=int, default=None)
    parser.add_argument('--hid_units', type=int, default=16)
    parser.add_argument('--hid_dimension', type=int, default=512)
    parser.add_argument('--num_experiments', type=int, default=1)
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but it is not available.")
    if args.device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    run_nonoverlapping(args)



