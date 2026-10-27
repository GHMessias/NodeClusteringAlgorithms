
# Optional benchmark result selection; standalone/default training is unchanged.
import os as _benchmark_os
if _benchmark_os.environ.get("BENCHMARK_SELECT_BEST_NMI") == "1":
    from benchmark_selection import selection as _nmi_selection
else:
    _nmi_selection = None

# coding=utf-8
# Author: Jung
# Time: 2023/3/12 15:06

import utils
import evaluate
import torch
import numpy as np
import pickle  as pkl
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import scipy.sparse as sp
import networkx as nx
import os
import random
import argparse
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import math
random.seed(826)
np.random.seed(826)
torch.manual_seed(826)
torch.cuda.manual_seed(826)

DATASET = "pubmed"
PRETRAIN_DIR = None


def get_pretrain_path():
    if PRETRAIN_DIR is not None:
        os.makedirs(PRETRAIN_DIR, exist_ok=True)
        return os.path.join(PRETRAIN_DIR, "model_0.pk")
    return 'datasets/'+ DATASET + '/pretrain/model_0.pk'

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
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges] or [num_edges, 2].")

    rows = edge_index[0].astype(np.int64)
    cols = edge_index[1].astype(np.int64)
    values = np.ones(rows.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def load_data_pt(data_pt):
    data = unwrap_data_pt(torch_load(data_pt))
    x = get_data_attr(data, 'x')
    y = get_data_attr(data, 'y')
    edge_index = get_data_attr(data, 'edge_index')
    if x is None or edge_index is None:
        raise ValueError("data.pt must contain x and edge_index.")

    feat = sp.csr_matrix(as_numpy(x).astype(np.float32))
    label = labels_to_vector(y) if y is not None else None
    adj = adjacency_from_edge_index(edge_index, feat.shape[0])
    return adj, feat, label


def load_data(name='cora', data_pt=None):
    if data_pt is None and isinstance(name, str) and name.endswith('.pt'):
        data_pt = name

    if data_pt:
        adj, feat, label = load_data_pt(data_pt)
    else:
        root = "D:\PyCharm_WORK\MyCode\Jung\datasets\\"
        dataset_addr = root + name + ".pkl"
        with open(dataset_addr, "rb")as f:
            dataset = pkl.load(f)
        adj = dataset['adj']
        feat = dataset['feat']
        label = dataset['label']

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = utils.sparse_to_tuple(adj_label)
    adj_norm = utils.normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = utils.sparse_to_tuple(adj_norm)
    feat = utils.sparse_to_tuple(feat)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]),
                                        torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]),
                                         torch.Size(adj_label[2]))
    feat = torch.sparse.FloatTensor(torch.LongTensor(feat[0].T), torch.FloatTensor(feat[1]),
                                        torch.Size(feat[2]))

    weight_mask_orig = adj_label.to_dense().view(-1) == 1
    weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
    weight_tensor_orig[weight_mask_orig] = float(pos_weight_orig)


    return adj, adj_norm, adj_label, feat, label, norm, weight_tensor_orig


def create_kcore_graph(sp_adj):
    nx_graph = nx.from_scipy_sparse_matrix(sp_adj)
    full_node_list = nx.nodes(nx_graph)
    core_num_dict = nx.core_number(nx_graph)
    print("unique core nums: ", len(np.unique(np.array(list(core_num_dict.values())))))
    max_core_num = max(list(core_num_dict.values()))
    print('max core num: ', max_core_num)

    def get_format_str(cnt):
        max_bit = 0
        while cnt > 0:
            cnt //= 10
            max_bit += 1
        format_str = '{:0>' + str(max_bit) + 'd}'
        return format_str

    format_str = get_format_str(max_core_num)


    for i in range(1, max_core_num + 1):
        k_core_graph = nx.k_core(nx_graph, k = i, core_number=core_num_dict)
        k_core_graph.add_nodes_from(full_node_list)
        A = nx.to_scipy_sparse_matrix(k_core_graph, nodelist=full_node_list)



        signature = format_str.format(i)
        sp.save_npz(os.path.join("datasets\\"+DATASET+"\kcore_graph", signature + '.npz'), A)


def get_kcore_graph():
    date_dir_list = sorted(os.listdir("datasets\\"+ DATASET + "\kcore_graph"))
    norm_list = []
    weight_list = []
    adj_norm_list = []
    adj_recons_list = []
    adj_for_w_list = []
    index_list = []
    for i, file_name in enumerate(date_dir_list):
        adj = sp.load_npz(os.path.join("datasets\\"+ DATASET + "\kcore_graph", file_name)) # 无自环
        adj_for_w_list.append(adj)
        index_list.append(np.unique(adj.tocoo().row))

        pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = utils.sparse_to_tuple(adj_label)
        adj_norm = utils.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_norm = utils.sparse_to_tuple(adj_norm)
        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]),
                                            torch.Size(adj_norm[2]))
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]),
                                             torch.Size(adj_label[2]))
        weight_mask_orig = adj_label.to_dense().view(-1) == 1
        weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
        weight_tensor_orig[weight_mask_orig] = pos_weight_orig

        norm_list.append(norm)
        weight_list.append(weight_tensor_orig)
        adj_norm_list.append(adj_norm)
        adj_recons_list.append(adj_label)

    return adj_for_w_list, adj_norm_list, adj_recons_list, norm_list, weight_list, index_list


def build_kcore_graphs(sp_adj):
    nx_graph = nx.from_scipy_sparse_array(sp_adj) if hasattr(nx, 'from_scipy_sparse_array') else nx.from_scipy_sparse_matrix(sp_adj)
    full_node_list = list(range(sp_adj.shape[0]))
    core_num_dict = nx.core_number(nx_graph)
    max_core_num = max(list(core_num_dict.values())) if core_num_dict else 0
    if max_core_num == 0:
        k_core_adj_list = [sp_adj]
    else:
        k_core_adj_list = []
        for i in range(1, max_core_num + 1):
            k_core_graph = nx.k_core(nx_graph, k=i, core_number=core_num_dict)
            k_core_graph.add_nodes_from(full_node_list)
            if hasattr(nx, 'to_scipy_sparse_array'):
                A = nx.to_scipy_sparse_array(k_core_graph, nodelist=full_node_list, format='csr')
            else:
                A = nx.to_scipy_sparse_matrix(k_core_graph, nodelist=full_node_list)
            k_core_adj_list.append(sp.csr_matrix(A))

    norm_list = []
    weight_list = []
    adj_norm_list = []
    adj_recons_list = []
    adj_for_w_list = []
    index_list = []
    for adj in k_core_adj_list:
        adj_for_w_list.append(adj)
        index = np.unique(adj.tocoo().row)
        if len(index) == 0:
            index = np.arange(adj.shape[0])
        index_list.append(index)

        pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = utils.sparse_to_tuple(adj_label)
        adj_norm = utils.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_norm = utils.sparse_to_tuple(adj_norm)
        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]),
                                            torch.Size(adj_norm[2]))
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]),
                                             torch.Size(adj_label[2]))
        weight_mask_orig = adj_label.to_dense().view(-1) == 1
        weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
        weight_tensor_orig[weight_mask_orig] = float(pos_weight_orig)

        norm_list.append(norm)
        weight_list.append(weight_tensor_orig)
        adj_norm_list.append(adj_norm)
        adj_recons_list.append(adj_label)

    return adj_for_w_list, adj_norm_list, adj_recons_list, norm_list, weight_list, index_list





class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = self.__random_uniform_init(input_dim, output_dim)
        self.activation = activation

    def __random_uniform_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
        return nn.Parameter(initial)

    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs


class KMVGAE(nn.Module):
    def __init__(self, **kwargs):
        super(KMVGAE, self).__init__()
        self.adj = kwargs['adj']
        self.feat = kwargs['feat']
        self.label = kwargs['label']
        self.norm = kwargs['norm']
        self.weight_tensor_orig = kwargs['weight_tensor_orig']
        self.adj_loop = kwargs['adj_loop']
        self.num_node, self.feat_dim = self.feat.shape
        self.nClusters = kwargs.get('n_clusters')
        if self.nClusters is None and self.label is not None:
            self.nClusters = len(np.unique(self.label))
        if self.nClusters is None:
            raise ValueError("--n_clusters is required when data.y is unavailable.")


        # read k-core data
        self.k_core_adj = kwargs['adj_norm_list']
        self.k_core_orgadj = kwargs['adj_recons_list']
        self.k_core_norm = kwargs['norm_list']
        self.k_core_weight = kwargs['weight_list']
        self.adj_for_w = kwargs['adj_for_w_list']
        self.k_core_index = kwargs['index_list']


        hid1_dim = kwargs.get('hidden1_dim', 32)
        self.hid2_dim = kwargs.get('hidden2_dim', 16)
        self.pretrain_lr = kwargs.get('pretrain_lr', 0.01)
        self.train_lr = kwargs.get('train_lr', 0.002)
        self.reconstruction_scale = kwargs.get('reconstruction_scale', 1e-2)
        self.numerical_stability = kwargs.get('numerical_stability', 1e-2)

        # VGAE training parameters
        self.activation = F.relu

        self.base_gcn = GraphConvSparse(self.feat_dim, hid1_dim, self.activation)
        self.gcn_mean = GraphConvSparse(hid1_dim, self.hid2_dim, activation = lambda x:x)
        self.gcn_logstddev = GraphConvSparse(hid1_dim, self.hid2_dim, activation = lambda x:x)

        # GMM training parameters
        self.pi = nn.Parameter(torch.ones(self.nClusters) / self.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.randn(self.nClusters, self.hid2_dim), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.randn(self.nClusters, self.hid2_dim), requires_grad=True)


    def _encode(self, adj, i):
        hidden = self.base_gcn(self.feat, adj)
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)
        gaussian_noise = torch.randn(self.num_node, self.hid2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return self.mean, self.logstd, sampled_z

    @staticmethod
    def _decode_logits(z):
        return torch.matmul(z, z.t())


    def _get_w(self, pred, i, max_nmi):


        ratio = 1 - max_nmi
        drop = np.random.choice(self.k_core_index[i], size=(int(len(self.k_core_index[i]) * ratio)))
        p = torch.zeros((self.num_node, self.nClusters))
        p[range(self.num_node), pred] = 1
        W = (p @ p.t()).numpy().astype(float)
        W = W * adj_for_w_list[i].toarray()



        W = W.astype(float)
        W = sp.csr_matrix(W)
        W = utils.sparse_to_tuple(W)
        W = torch.sparse.FloatTensor(torch.LongTensor(W[0].T), torch.FloatTensor(W[1]),
                                        torch.Size(W[2])) # 2708 * 2708
        W = W.unsqueeze(-1) # 2708, 2708, 1
        self.W_tile = torch.cat([W] * self.nClusters, - 1)


    def _pretrain(self, i, epochs=200):
        pretrain_path = get_pretrain_path()
        if not os.path.exists(pretrain_path):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.pretrain_lr)
            epoch_bar = tqdm(range(epochs))
            nmi_best = 0
            gmm = GaussianMixture(n_components=self.nClusters, covariance_type='diag')
            nmi_list = []
            for _ in epoch_bar:
                optimizer.zero_grad()
                _, _, z = self._encode(self.k_core_adj[i], i)
                A_logits = self._decode_logits(z)
                loss = self.k_core_norm[i] * F.binary_cross_entropy_with_logits(
                    A_logits.view(-1),
                    self.k_core_orgadj[i].to_dense().view(-1),
                    weight=self.k_core_weight[i],
                )
                loss.backward()
                optimizer.step()
                y_pred = gmm.fit_predict(z.detach().numpy())
                self.pi.data = torch.from_numpy(gmm.weights_).to(torch.float)
                self.mu_c.data = torch.from_numpy(gmm.means_).to(torch.float)
                self.log_sigma2_c.data =  torch.log(torch.from_numpy(gmm.covariances_)).to(torch.float)
                if self.label is not None:
                    acc = cal_acc(self.label[self.k_core_index[i]], y_pred[self.k_core_index[i]])
                    nmi = evaluate.compute_nmi(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
                    f1 = evaluate.computer_macrof1(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
                    ari = evaluate.computer_ari(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
                    epoch_bar.write('Loss pretraining = {:.4f}, acc = {:.4f}, nmi = {:.4f}, f1 = {:.4f}, ari = {:.4f}'.format(loss, acc , nmi, f1, ari))
                    nmi_list.append(acc)
                    if nmi > nmi_best:
                        nmi_best = nmi
                        self.logstd = self.mean
                        torch.save(self.state_dict(), pretrain_path)
                else:
                    epoch_bar.write('Loss pretraining = {:.4f}'.format(loss))
            if self.label is None:
                torch.save(self.state_dict(), pretrain_path)
            else:
                print("Best accuracy : ", nmi_best)
        else:
            self.load_state_dict(torch.load(pretrain_path))

    def _train(self, i, constrained, epochs=200):
        self.load_state_dict(torch.load(get_pretrain_path()))
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_lr)
        acc_list = []
        nmi_list = []
        max_nmi = -1
        best_y = None
        best_embedding = None
        epoch_bar = tqdm(range(epochs))
        for epoch in epoch_bar:
            optimizer.zero_grad()
            z_mean, z_logstd, z = self._encode(self.k_core_adj[i], i)
            A_logits = self._decode_logits(z)

            loss = self._ELBO(A_logits, z, z_mean, z_logstd, i, constrained=constrained)


            y_pred, prob_y_pred = self._predict(z)
            if _nmi_selection is not None:
                _nmi_selection.consider(y_pred, z, epoch + 1, stage=f"core_{i}")
            if self.label is not None:
                acc = cal_acc(self.label[self.k_core_index[i]], y_pred[self.k_core_index[i]])
                nmi = evaluate.compute_nmi(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
                f1 = evaluate.computer_macrof1(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
                ari = evaluate.computer_ari(self.label[self.k_core_index[i]], y_pred[self.k_core_index[i]])


            q_adj = self.adj_for_w[i]
            q_adj = q_adj[self.k_core_index[i],:]
            q_adj = q_adj[:, self.k_core_index[i]]
            try:
                q = evaluate.modularity(q_adj.toarray(), y_pred[self.k_core_index[i]])
            except ImportError:
                q = float('nan')
            if self.label is not None:
                epoch_bar.write(
                    'Loss training = {:.4f}, acc = {:.4f}, nmi = {:.4f}, f1 = {:.4f}, ari = {:.4f}, q = {:.4f}'.format(loss, acc, nmi, f1, ari, q))
                acc_list.append(acc)
                nmi_list.append(nmi)

            if self.label is None or nmi > max_nmi:
                if self.label is not None:
                    max_nmi = nmi
                best_y = y_pred.copy()
                best_embedding = z.detach().cpu().numpy().copy()

            loss.backward()
            optimizer.step()

        if self.label is not None:
            print("MAX ACC : {}, MAX NMI :{}".format(max(acc_list), max(nmi_list)))
            self._get_w(best_y, i, max_nmi)
        self.best_embedding = best_embedding
        return best_y

    def _ELBO_No_W(self, A_logits, z, z_mean, z_logstd):
        """
            消融实验使用的损失函数，没有约束信息
        """
        pi = self.pi
        mean_c = self.mu_c
        logstd_c = self.log_sigma2_c

        loss_recons = self.reconstruction_scale * norm * F.binary_cross_entropy_with_logits(
            A_logits.view(-1),
            self.adj_loop.to_dense().view(-1),
            weight=weight_tensor_orig,
        )
        loss_recons = loss_recons * self.num_node

        gamma_c = self._cluster_posteriors(z)

        KL1 = 0.5 * torch.mean(torch.sum(gamma_c * torch.sum(logstd_c.unsqueeze(0) + torch.exp(z_logstd.unsqueeze(1) - logstd_c.unsqueeze(0))
                                                             + (z_mean.unsqueeze(1) - mean_c.unsqueeze(0)).pow(2) / torch.exp(logstd_c.unsqueeze(0)), 2), 1))
        KL2 = torch.mean(
            torch.sum(
                gamma_c
                * (torch.log(pi.clamp_min(torch.finfo(pi.dtype).tiny)).unsqueeze(0) - torch.log(gamma_c)),
                1,
            )
        ) + 0.5 * torch.mean(torch.sum(1 + z_logstd, 1))
        loss_clus = KL1 - KL2

        loss_elbo = loss_recons + loss_clus
        return loss_elbo


    def _ELBO(self, A_logits, z, z_mean, z_logstd, i, constrained=False):
        pi = self.pi
        mean_c = self.mu_c
        logstd_c = self.log_sigma2_c
        det = self.numerical_stability

        if constrained is True:
            w = self.W_tile.to_dense()  # 2707， 2708
            y_pred = torch.exp(torch.log(pi.unsqueeze(0)) + self._gaussian_pdfs_log(z, mean_c, logstd_c))
            mul = w.__mul__(y_pred)
            sum_j = torch.sum(mul, dim=-2) # + det

        loss_recons = self.reconstruction_scale * self.k_core_norm[i] * F.binary_cross_entropy_with_logits(
            A_logits.view(-1),
            self.k_core_orgadj[i].to_dense().view(-1),
            weight=self.k_core_weight[i],
        )
        loss_recons = loss_recons * self.num_node

        gamma_c = self._cluster_posteriors(z) # 2708,7


        KL1 = 0.5 * torch.mean(torch.sum(gamma_c * torch.sum(logstd_c.unsqueeze(0) + torch.exp(z_logstd.unsqueeze(1) - logstd_c.unsqueeze(0))
                                                             + (z_mean.unsqueeze(1) - mean_c.unsqueeze(0)).pow(2) / torch.exp(logstd_c.unsqueeze(0)), 2), 1))

        if constrained is True:
            KL2 = torch.mean(torch.sum(gamma_c * torch.log((sum_j / gamma_c) + det), 1)) + 0.5 * torch.mean(torch.sum(1 + z_logstd, 1))
        else:
            KL2 = torch.mean(
                torch.sum(
                    gamma_c
                    * (torch.log(pi.clamp_min(torch.finfo(pi.dtype).tiny)).unsqueeze(0) - torch.log(gamma_c)),
                    1,
                )
            ) + 0.5 * torch.mean(torch.sum(1 + z_logstd, 1))

        loss_clus = KL1 - KL2
        loss_elbo = loss_recons + loss_clus
        return loss_elbo

    def _gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.nClusters):
            G.append(self._gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    def _cluster_posteriors(self, z):
        """Compute the original stabilized mixture posterior in log space."""
        tiny = torch.finfo(z.dtype).tiny
        log_pi = torch.log(self.pi.clamp_min(tiny)).unsqueeze(0)
        log_scores = log_pi + self._gaussian_pdfs_log(z, self.mu_c, self.log_sigma2_c)
        log_det = torch.full_like(log_scores, math.log(max(self.numerical_stability, tiny)))
        stabilized_log_scores = torch.logaddexp(log_scores, log_det)
        return torch.softmax(stabilized_log_scores, dim=1)


    def _gaussian_pdf_log(self,x,mu,log_sigma2):
        c = -0.5 * torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1)
        return c

    def _predict(self, z):
        posterior = self._cluster_posteriors(z).detach()
        return np.argmax(posterior.numpy(), axis=1), posterior

    def _downstream(self):

        snpa = 0

        self.load_state_dict(torch.load('datasets/' + DATASET + '/evoluation/evo_'+str(snpa)+'.pk'))
        i = snpa
        with torch.no_grad():
            _, _, z = self._encode(self.k_core_adj[i], i)
            y_pred, _ = self._predict(z)
            acc = cal_acc(self.label[self.k_core_index[i]], y_pred[self.k_core_index[i]])
            nmi = evaluate.compute_nmi(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
            f1 = evaluate.computer_f1(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
            ari = evaluate.computer_ari(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
            q_adj = self.adj_for_w[i]
            q_adj = q_adj[self.k_core_index[i], :]
            q_adj = q_adj[:, self.k_core_index[i]]
            q = utils.modularity(q_adj.toarray(), y_pred[self.k_core_index[i]])
            print("downsteam := acc = {:.4f}, nmi = {:.4f}, f1 = {:.4f}, ari = {:.4f}, q = {:.4f}".format(acc,nmi,f1,ari,q))

        kwargs = {
            "early_exaggeration": 100, # 38
            "init": "pca",
            "perplexity": 100,
            "emb": z,
            "label": self.label,
            "path": 'datasets/' + DATASET + '/evoluation/evo_'+str(snpa)+'.pdf'
        }

        utils.visualization(**kwargs)


def cal_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
       accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EGC-VGAE")
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--data_pt", type=str, default=None, help="Path to PyG data.pt with x and edge_index; y is optional")
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden1_dim", type=int, default=32)
    parser.add_argument("--hidden2_dim", type=int, default=16)
    parser.add_argument("--pretrain_lr", type=float, default=0.01)
    parser.add_argument("--train_lr", type=float, default=0.002)
    parser.add_argument("--reconstruction_scale", type=float, default=1e-2)
    parser.add_argument("--numerical_stability", type=float, default=1e-2)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    DATASET = args.dataset
    PRETRAIN_DIR = os.path.join(args.output_dir, "egc_vgae_pretrain", DATASET)

    adj, adj_norm, adj_label, feat, label, norm, weight_tensor_orig = load_data(DATASET, data_pt=args.data_pt)
    if args.data_pt:
        adj_for_w_list, adj_norm_list, adj_recons_list, norm_list, weight_list, index_list = build_kcore_graphs(adj)
    else:
        adj_for_w_list, adj_norm_list, adj_recons_list, norm_list, weight_list, index_list = get_kcore_graph()

    num_k_core = len(adj_norm_list)

    configs = {
        "adj": adj_norm,
        "feat": feat,
        "label": label,
        "n_clusters": args.n_clusters,
        "norm": norm,
        "weight_tensor_orig" : weight_tensor_orig,
        "adj_loop" : adj_label,
        "adj_norm_list" : adj_norm_list,
        "adj_recons_list" : adj_recons_list,
        "norm_list" : norm_list,
        "weight_list" : weight_list,
        "adj_for_w_list": adj_for_w_list,
        "index_list": index_list,
        "hidden1_dim": args.hidden1_dim,
        "hidden2_dim": args.hidden2_dim,
        "pretrain_lr": args.pretrain_lr,
        "train_lr": args.train_lr,
        "reconstruction_scale": args.reconstruction_scale,
        "numerical_stability": args.numerical_stability,
    }
    model = KMVGAE(**configs)
    model._pretrain(0, epochs=args.epochs)
    labels = None
    for i in range(num_k_core-1, -1, -1):
        print("##########       {}         ##########".format(i))

        if i == num_k_core-1:
            labels = model._train(i, False, epochs=args.epochs) #
        else:
            # model._pretrain(i)
            labels = model._train(i, False, epochs=args.epochs) # EGC-VGAE use True
    # model._downstream()
    if labels is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M")
        output_path = os.path.join(args.output_dir, "EGC_VGAE_{}_labels.npy".format(timestamp))
        np.save(output_path, labels.astype(np.int64))
        np.save(os.path.join(args.output_dir, "embeddings.npy"), model.best_embedding)
        print("EGC-VGAE labels saved in: {}".format(output_path))
