import numpy as np
import torch
from sklearn.preprocessing import normalize

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


def get_dataset(dataset):
    # datasets = Planetoid('./dataset', dataset)
    datasets = Planetoid(root='./dataset', name=dataset)
    return datasets


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_data_attr(data, name):
    if isinstance(data, dict):
        return data.get(name)
    return getattr(data, name, None)


def unwrap_data_pt(obj):
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if get_data_attr(item, "x") is not None and get_data_attr(item, "edge_index") is not None:
                return item
    return obj


def as_tensor(value):
    if hasattr(value, "is_sparse") and value.is_sparse:
        value = value.to_dense()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def normalize_edge_index(edge_index, num_nodes):
    edge_index = as_tensor(edge_index).long()
    if edge_index.dim() != 2:
        raise ValueError("edge_index must be a 2D tensor.")
    if edge_index.shape[0] != 2:
        edge_index = edge_index.t()
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges] or [num_edges, 2].")

    src, dst = edge_index[0], edge_index[1]
    edge_index = torch.cat([
        torch.stack([src, dst], dim=0),
        torch.stack([dst, src], dim=0),
    ], dim=1)
    valid = (
        (edge_index[0] >= 0)
        & (edge_index[0] < num_nodes)
        & (edge_index[1] >= 0)
        & (edge_index[1] < num_nodes)
        & (edge_index[0] != edge_index[1])
    )
    edge_index = edge_index[:, valid]
    return torch.unique(edge_index, dim=1).contiguous()


def load_data_pt(data_pt):
    data = unwrap_data_pt(torch_load(data_pt))
    x = get_data_attr(data, "x")
    y = get_data_attr(data, "y")
    edge_index = get_data_attr(data, "edge_index")
    if x is None or y is None or edge_index is None:
        raise ValueError("data.pt must contain x, y and edge_index.")

    x = as_tensor(x).float()
    y = as_tensor(y)
    if y.dim() > 1:
        y = torch.argmax(y, dim=1)
    y = y.long().view(-1)
    edge_index = normalize_edge_index(edge_index, x.shape[0])
    return Data(x=x, y=y, edge_index=edge_index)


def data_preprocessing(dataset):
    if isinstance(dataset, str):
        dataset = load_data_pt(dataset)

    dataset.x = as_tensor(dataset.x).float()
    dataset.y = as_tensor(dataset.y).long().view(-1)
    dataset.edge_index = normalize_edge_index(dataset.edge_index, dataset.x.shape[0])

    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj.clone()

    dataset.adj += torch.eye(dataset.x.shape[0])
    dataset.adj = normalize(dataset.adj.cpu().numpy(), norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    return dataset

def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)

