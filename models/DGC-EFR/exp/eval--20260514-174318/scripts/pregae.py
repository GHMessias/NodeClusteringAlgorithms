from __future__ import print_function, division
import argparse
import os
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from utils import load_data, load_graph, sparse_mx_to_torch_sparse_tensor
from GNN import GraphAttentionLayer
from evaluation import eva

class DataPtDataset(Dataset):
    def __init__(self, x, y, edge_index):
        self.x = x
        self.y = y
        self.edge_index = edge_index

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), \
               torch.from_numpy(np.array(self.y[idx])), \
               torch.from_numpy(np.array(idx))

def torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')

def as_numpy(value):
    if hasattr(value, 'is_sparse') and value.is_sparse:
        value = value.to_dense()
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)

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

def load_data_pt(path):
    data = unwrap_data_pt(torch_load(path))
    x = get_data_attr(data, 'x')
    y = get_data_attr(data, 'y')
    edge_index = get_data_attr(data, 'edge_index')

    if x is None or y is None or edge_index is None:
        raise ValueError("Expected data.pt with x, y and edge_index attributes.")

    x = as_numpy(x).astype(np.float32)
    y = as_numpy(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    elif y.ndim > 1:
        y = np.argmax(y, axis=1)
    y = y.astype(int)

    edge_index = as_numpy(edge_index).astype(np.int64)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    return DataPtDataset(x, y, edge_index)

def row_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)

def load_graph_from_edge_index(edge_index, num_nodes):
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )

    adj_noeye = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj_noeye + sp.eye(adj_noeye.shape[0])
    adj = row_normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_label = sparse_mx_to_torch_sparse_tensor(adj_noeye)
    return adj, adj_label

# the graph autoencoder
class GAE(nn.Module):
    def __init__(self, num_features, hidden1_size, hidden2_size, embedding_size, alpha):
        super(GAE, self).__init__()
        self.embedding_size = embedding_size
        self.alpha = alpha
        # encoder configuration
        self.conv1 = GraphAttentionLayer(num_features, hidden1_size, alpha)
        self.conv2 = GraphAttentionLayer(hidden1_size, hidden2_size, alpha)
        self.conv3 = GraphAttentionLayer(hidden2_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        h = self.conv3(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = dot_product_decode(z)
        return A_pred,z


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred


def pretrain_gae(dataset, args):
    model = GAE(num_features=args.input_dim, hidden1_size=args.hidden1_dim,  hidden2_size=args.hidden2_dim,
                  embedding_size=args.hidden3_dim,alpha=args.alpha).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    if hasattr(dataset, 'edge_index'):
        adj, adj_label = load_graph_from_edge_index(dataset.edge_index, dataset.x.shape[0])
    else:
        adj, adj_label = load_graph(args.name, args.k)
    adj_dense = adj.to_dense()
    adj_numpy = adj_dense.data.cpu().numpy()
    t=10
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    M = torch.Tensor(M_numpy).to(device)

    adj = adj_dense.to(device)
    adj_label = adj_label.to(device)

    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    for epoch in range(args.epochs):
        model.train()
        A_pred,z = model(data, adj, M)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            _,z = model(data, adj, M)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
            eva(y, kmeans.labels_, epoch)

    output_dir = os.path.dirname(args.pregae_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), args.pregae_path)
    print("Saved pre-trained GAE: {}".format(args.pregae_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='acm')
    parser.add_argument('--data_pt', type=str, default=None, help='Optional PyG data.pt path with x, y and edge_index')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--k', type=int, default=5) 
    parser.add_argument('--n_clusters', default=None, type=int)
    parser.add_argument('--input_dim', default=None, type=int)
    parser.add_argument('--hidden1_dim', default=1024, type=int)
    parser.add_argument('--hidden2_dim', default=256, type=int)
    parser.add_argument('--hidden3_dim', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--pregae_path', type=str, default=None)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.data_pt:
        dataset = load_data_pt(args.data_pt)
    else:
        dataset = load_data(args.name)

    if args.name == 'acm' and not args.data_pt:
        args.k = None
        if args.n_clusters is None:
            args.n_clusters = 3

    if args.name == 'dblp' and not args.data_pt:
        args.k = None
        if args.n_clusters is None:
            args.n_clusters = 4

    if args.name == 'cora' and not args.data_pt:
        args.k = None
        if args.n_clusters is None:
            args.n_clusters = 7

    if args.name == 'cite' and not args.data_pt:
        args.k = None
        if args.n_clusters is None:
            args.n_clusters = 6

    if args.input_dim is None:
        args.input_dim = dataset.x.shape[1]
    if args.n_clusters is None:
        args.n_clusters = len(set(np.asarray(dataset.y).reshape(-1).tolist()))
    if args.pregae_path is None:
        args.pregae_path = 'data/pregae_{}.pkl'.format(args.name)


    print(args)
    pretrain_gae(dataset, args)
