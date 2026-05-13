import os

import numpy as np
import scipy.sparse as sp
import h5py
import torch
from torch.utils.data import Dataset


def _torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _is_data_pt(path):
    return isinstance(path, (str, os.PathLike)) and str(path).endswith('.pt')


def _unwrap_data_pt(obj):
    if isinstance(obj, (tuple, list)):
        for item in obj:
            if hasattr(item, 'x') or (isinstance(item, dict) and 'x' in item):
                return item
        return obj[0]
    if isinstance(obj, dict):
        if 'data' in obj:
            return obj['data']
        if 'x' in obj:
            return obj
    return obj


def _get_attr(data, name):
    if isinstance(data, dict):
        return data.get(name)
    return getattr(data, name, None)


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.is_sparse:
            value = value.to_dense()
        return value.detach().cpu().numpy()
    if sp.issparse(value):
        return value.toarray()
    return np.asarray(value)


def _load_data_pt(data_pt):
    data = _unwrap_data_pt(_torch_load(data_pt))
    x = _to_numpy(_get_attr(data, 'x'))
    y = _to_numpy(_get_attr(data, 'y'))
    edge_index = _to_numpy(_get_attr(data, 'edge_index'))

    if x is None:
        raise ValueError(f'{data_pt} precisa conter data.x')
    if y is None:
        raise ValueError(f'{data_pt} precisa conter data.y')
    if edge_index is None:
        raise ValueError(f'{data_pt} precisa conter data.edge_index')

    x = x.astype(np.float32)
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.argmax(axis=-1)
    y = y.reshape(-1).astype(np.int64)
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2:
        raise ValueError('data.edge_index precisa ser uma matriz 2D')
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T
    if edge_index.shape[0] != 2:
        raise ValueError('data.edge_index precisa ter shape [2, num_edges]')

    return x, y, edge_index


def _edge_index_to_adj(edge_index, num_nodes):
    rows, cols = edge_index
    mask = (rows >= 0) & (rows < num_nodes) & (cols >= 0) & (cols < num_nodes)
    rows = rows[mask]
    cols = cols[mask]
    adj = sp.coo_matrix(
        (np.ones(rows.shape[0], dtype=np.float32), (rows, cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    return sparse_mx_to_torch_sparse_tensor(adj)


def load_graph(data_path, dataset, k, data_pt=None):
    if data_pt is not None or _is_data_pt(data_path):
        pt_path = data_pt if data_pt is not None else data_path
        x, _, edge_index = _load_data_pt(pt_path)
        return _edge_index_to_adj(edge_index, x.shape[0])

    if k:
        path = f'{data_path}/graph/{dataset}{k}_graph.txt' 
    else:
        path = f'{data_path}/graph/{dataset}_graph.txt' 

    data = np.loadtxt(f'{data_path}/data/{dataset}.txt')
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, check_invariants=False).coalesce()


class load_data(Dataset):
    def __init__(self, data_path=None, dataset=None, data_pt=None):
        if data_pt is not None or _is_data_pt(data_path):
            pt_path = data_pt if data_pt is not None else data_path
            self.x, self.y, self.edge_index = _load_data_pt(pt_path)
            return

        self.x = np.loadtxt(f'{data_path}/data/{dataset}.txt', dtype=float)
        self.y = np.loadtxt(f'{data_path}/data/{dataset}_label.txt', dtype=int)
        self.edge_index = None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))
