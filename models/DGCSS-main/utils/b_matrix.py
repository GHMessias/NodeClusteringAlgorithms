import numpy as np
import torch
from numpy.linalg import matrix_power
from torch_geometric.data import Data


class BMatrix:
    def __init__(self, data):
        self.n_nodes = len(data.x)
        self.adj_matrix = None
        self.edge_index = None
        self.b_matrix = None

    def calc_t_order_neighbors(self, data, t=1):
        """
        Calculate the T order neighbors for the given adjancency matrix.
        """

        self.__create_adj_matrix(data)

        b = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.adj_matrix[i][j] == 0:
                    b[i][j] = 0
                    continue
                sum_b_row = np.sum(self.adj_matrix[i])
                b[i][j] = 1 / sum_b_row

        b_sum = b
        for i in range(1, t):
            b_sum = b_sum + matrix_power(b, i)

        self.b_matrix = b_sum / t

    def create_edge_index(self):
        self.__adj_to_edge_index()
        self.__calculate_weights()

    def __create_adj_matrix(self, data):
        new_adj_matrix = np.zeros((data.num_nodes, data.num_nodes))

        src = data.edge_index[0].detach().numpy()
        tgt = data.edge_index[1].detach().numpy()

        for i in range(len(src)):
            new_adj_matrix[src[i]][tgt[i]] = 1

        self.adj_matrix = new_adj_matrix

    def __adj_to_edge_index(self):
        """
        Transform a adjacency matrix to the Edge Index format.
        """
        adj_t = torch.tensor(self.adj_matrix)
        self.edge_index = adj_t.nonzero().t().contiguous()

    def __calculate_weights(self):
        """
        Add weights to B matrix edge_index format, based on the
        Adjacency matrix values.
        """
        b_weights = []

        for i in range(len(self.edge_index[0])):
            b_weights.append(
                self.adj_matrix[self.edge_index[0][i], self.edge_index[1][i]]
            )

        b_weights = torch.tensor(b_weights)

        new_edge_index = Data(edge_index=self.edge_index, edge_attr=b_weights)

        self.edge_index = new_edge_index

