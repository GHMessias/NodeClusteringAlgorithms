from __future__ import print_function, division
import argparse
import os
import sys
import time
import glob
import logging
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import load_data, load_graph, create_exp_dir, sparse_mx_to_torch_sparse_tensor
from GNN import GraphAttentionLayer
from evaluation import eva

# The basic autoencoder
class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3,  n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z,  v=1):
        super(AE, self).__init__()

        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)
        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

        # degree
        self.v = v

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z

# graph autoencoder
class GAE(nn.Module):
    def __init__(self, num_features, hidden1_size, hidden2_size, embedding_size, alpha):
        super(GAE, self).__init__()
        # encoder
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GraphAttentionLayer(num_features, hidden1_size, alpha)
        self.conv2 = GraphAttentionLayer(hidden1_size, hidden2_size, alpha)
        self.conv3 = GraphAttentionLayer(hidden2_size, embedding_size, alpha)

    def forward(self, x, adj, M, enc_feat1, enc_feat2, enc_feat3):
        sigma = 0.1
        h = self.conv1(x, adj, M)
        h = self.conv2((1-sigma)*h + sigma*enc_feat1, adj, M)
        h = self.conv3((1-sigma)*h + sigma*enc_feat2, adj, M)
        z = F.normalize((1 -sigma)*h + sigma*enc_feat3, p=2, dim=1)
        # decoder
        A_pred = dot_product_decode(z)
        return A_pred, z


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class DataPtDataset(torch.utils.data.Dataset):
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

class DGCEFR(nn.Module):
    def __init__(self, num_features, hidden1_size, hidden2_size, embedding_size, alpha, num_clusters, n_z, v=1):
        super(DGCEFR, self).__init__()
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.num_clusters = num_clusters
        self.v = v

        # pre-trained ae
        self.pre_ae =  AE(hidden1_size, hidden2_size, embedding_size, embedding_size, hidden2_size, hidden1_size, num_features, n_z = n_z)
        self.pre_ae.load_state_dict(torch.load(args.preae_path, map_location='cpu'))
        # pre-trained gae
        self.pre_gae = GAE(num_features, hidden1_size, hidden2_size, embedding_size, alpha)
        self.pre_gae.load_state_dict(torch.load(args.pregae_path, map_location='cpu'))

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, x, adj, M):
        x_bar, enc_h1, enc_h2, enc_h3, z = self.pre_ae(x)
        A_pred, z = self.pre_gae(x, adj, M, enc_h1, enc_h2, enc_h3)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return A_pred, z, q, x_bar


def train(dataset, args):
    # load initialized model
    model = DGCEFR(num_features=args.input_dim, hidden1_size=args.hidden1_dim, hidden2_size=args.hidden2_dim,
                  embedding_size=args.hidden3_dim, alpha=args.alpha, num_clusters=args.n_clusters, n_z=args.n_z ).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    # adjacent matrix
    if hasattr(dataset, 'edge_index'):
        adj, adj_label = load_graph_from_edge_index(dataset.edge_index, dataset.x.shape[0])
    else:
        adj, adj_label = load_graph(args.name, args.k)
    adj_dense = adj.to_dense()
    adj_numpy = adj_dense.data.cpu().numpy()
    t =10
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    M = torch.Tensor(M_numpy).to(device)

    adj = adj_dense.to(device)
    adj_label = adj_label.to(device)

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    with torch.no_grad():
         _, z,_, _ = model(data, adj, M)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    acc, nmi, ari, f1 = eva(y, y_pred, 'pae')
    logging.info('epoch %d acc %.4f nmi %.4f ari %.4f f1 %.4f', 0, acc, nmi, ari, f1)
    acc_best, nmi_best, ari_best, f1_best, epoch_best = 0.0, 0.0, 0.0, 0.0, 0
    labels_best = y_pred.astype(np.int64)
    for epoch in range(args.epochs):
        model.train()
        if epoch % args.update_interval == 0:
            # update_interval
            A_pred, z, tmp_q, x_bar = model(data, adj, M)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res2 = p.data.cpu().numpy().argmax(1)  # P

            acc, nmi, ari, f1 = eva(y, res2, str(epoch) + 'P')
            logging.info('epoch %s acc %.4f nmi %.4f ari %.4f f1 %.4f', str(epoch) + ' P', acc, nmi, ari, f1)
            acc, nmi, ari, f1 = eva(y, res1, str(epoch) + 'Q')
            logging.info('epoch %s acc %.4f nmi %.4f ari %.4f f1 %.4f', str(epoch) + ' Q', acc, nmi, ari, f1)
            if acc > acc_best:
                acc_best, nmi_best, ari_best, f1_best = acc, nmi, ari, f1
                epoch_best = epoch
                labels_best = res1.astype(np.int64)

        A_pred, z, q, x_bar = model(data, adj, M)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss_gae = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))
        re_loss_ae = F.mse_loss(x_bar, data)
        # loss function with 3 parts
        loss = args.lambda1 * kl_loss + 1.0*re_loss_gae + 10*re_loss_ae

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(epoch_best, ':acc_best {:.4f}'.format(acc_best), ', nmi {:.4f}'.format(nmi_best), ', ari {:.4f}'.format(ari_best),
            ', f1 {:.4f}'.format(f1_best))
    output_path = os.path.join(args.output_dir, "DGC_EFR_{}_labels.npy".format(args.name))
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(output_path, labels_best)
    print("DGC-EFR labels saved in: {}".format(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='acm')
    parser.add_argument('--data_pt', type=str, default=None, help='Optional PyG data.pt path with x, y and edge_index')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--k',type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200 )
    parser.add_argument('--lambda1', type=float, default=1 )
    parser.add_argument('--n_clusters', default=None, type=int)
    parser.add_argument('--input_dim', default=None, type=int)
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--hidden1_dim', default=1024, type=int)
    parser.add_argument('--hidden2_dim', default=256, type=int)
    parser.add_argument('--hidden3_dim', default=16, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--preae_path', type=str, default=None)
    parser.add_argument('--pregae_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.save = './exp/eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # load data
    if args.data_pt:
        dataset = load_data_pt(args.data_pt)
    else:
        dataset = load_data(args.name)

    if args.input_dim is None:
        args.input_dim = dataset.x.shape[1]
    if args.n_clusters is None:
        args.n_clusters = len(set(np.asarray(dataset.y).reshape(-1).tolist()))

    # load pre-trained models
    if args.pregae_path is None:
        args.pregae_path = 'data/pregae_{}.pkl'.format(args.name)
    if args.preae_path is None:
        args.preae_path = 'data/preae_{}.pkl'.format(args.name)
    print(args)
    train(dataset, args)
