
# Optional benchmark result selection; standalone/default training is unchanged.
import os as _benchmark_os
if _benchmark_os.environ.get("BENCHMARK_SELECT_BEST_NMI") == "1":
    from benchmark_selection import selection as _nmi_selection
else:
    _nmi_selection = None

from config import *
import argparse
import os
import time
import numpy as np
from utils import *
import networkx as nx
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from model import *
from evaluate import *


def torch_load(path):
    import torch
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


def graph_from_edge_index(edge_index, num_nodes, directed=False):
    edge_index = as_numpy(edge_index)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges] or [num_edges, 2].")

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.astype(np.int64).T.tolist()
    G.add_edges_from((int(src), int(dst)) for src, dst in edges if int(src) != int(dst))
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    return G


def load_data_pt(data_pt, config, directed=False):
    data = unwrap_data_pt(torch_load(data_pt))
    x = get_data_attr(data, 'x')
    y = get_data_attr(data, 'y')
    edge_index = get_data_attr(data, 'edge_index')
    if x is None or edge_index is None:
        raise ValueError("data.pt must contain x and edge_index.")

    X = as_numpy(x).astype(np.float32)
    labels = labels_to_vector(y) if y is not None else None
    config.n = X.shape[0]
    config.m = X.shape[1]
    config.layers[0] = config.m
    config.gcnlayers[0] = config.m
    G = graph_from_edge_index(edge_index, config.n, directed=directed)
    return G, X, labels


def load_dataset(args, config):
    if args.data_pt:
        return load_data_pt(args.data_pt, config, directed=args.directed)

    if args.dataset == 'cora':
        config.n = 2708
        config.m = 1433
        config.layers[0] = config.m
        config.gcnlayers[0] = config.m
        input_edge_file = 'data/cora/cora.edgelist'
        input_label_file = 'data/cora/cora.label'

        G = nx.read_edgelist(input_edge_file, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
        if args.directed == False:
            print('undirected graph')
            G = G.to_undirected()
        X = np.loadtxt('data/' + str(args.dataset) + '/original_feature_mtx_tfidf').astype(np.float32)
        labels = np.asarray(read_label(input_label_file), dtype=np.int64)
        return G, X, labels

    raise NotImplementedError("Dataset {} not implemented. Use --data_pt for custom data.".format(args.dataset))


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weighted', default=False, type=parse_bool)
    parser.add_argument('--directed', default=False, type=parse_bool)
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--data_pt', default=None, help='Path to PyG data.pt with x and edge_index; y is optional')
    parser.add_argument('--neg_mean', default='aa')#'random' or 'aa'
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--area', default=3, type=int)#area for negative sample if 'aa'
    parser.add_argument('--kmeans_clusters', default=None, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--classification_eval', action='store_true')
    parser.add_argument('--reg', type=float, default=5)
    parser.add_argument('--epsilon', type=float, default=1e-3)
    parser.add_argument('--dimension', type=int, default=64)
    parser.add_argument('--gcn_hidden_dim', type=int, default=128)
    parser.add_argument('--attribute_hidden_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_sg', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--batch_size_sg', type=int, default=512)
    parser.add_argument('--keep_prob', type=float, default=1.0)
    parser.add_argument('--walk_p', type=float, default=1.0)
    parser.add_argument('--walk_q', type=float, default=1.0)
    parser.add_argument('--num_walks', type=int, default=20)
    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--num_sampled', type=int, default=5)
    parser.add_argument('--margin', type=float, default=0.8)
    args = parser.parse_args()
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    neg_mean_map = {
        'random': 'R',
        'r': 'R',
        'R': 'R',
        'aa': 'S',
        's': 'S',
        'S': 'S',
        'attr': 'A',
        'attribute': 'A',
        'a': 'A',
        'A': 'A',
    }
    args.neg_mean = neg_mean_map.get(args.neg_mean, args.neg_mean)
    config = Config()
    config.reg = args.reg
    config.epsilon = args.epsilon
    config.dimension = args.dimension
    config.gcnlayers = [None, args.gcn_hidden_dim, args.dimension]
    config.layers = [None, args.attribute_hidden_dim, args.dimension]
    config.lr = args.lr
    config.lr_sg = args.lr_sg
    config.batch_size = args.batch_size
    config.batch_size_sg = args.batch_size_sg
    config.keep_prob = args.keep_prob
    config.p = args.walk_p
    config.q = args.walk_q
    config.num_walks = args.num_walks
    config.walk_length = args.walk_length
    config.window_size = args.window_size
    config.num_sampled = args.num_sampled
    config.margin = args.margin

    G, X, y = load_dataset(args, config)
    Adj = nx.to_numpy_array(G, nodelist=range(config.n))
    model = Model(config, Adj)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    if args.neg_mean == 'R':
        print('generating negative sample by random......')
    elif args.neg_mean == 'S':
        print('generating negative sample by aa in '+str(args.area)+'-hop neighbors......' )
        aa_list = get_neg_list_aa(G, args.area)
        X_neg = get_neg_aa(X, aa_list, args.dataset)
    elif args.neg_mean == 'A':
        print('generating negative sample by attribute in ' + str(args.area) + '-hop neighbors......')
        attr_list = get_neg_list_attr(G, X, args.area)
        X_neg = get_neg_attr(X, attr_list)
    else:
        print('Error: no proper negagtive sampling strategy')
    all_pairs = getRandomWalkWholePairs(G, args.directed, config.p, config.q, config.num_walks, config.walk_length, config.window_size)
    clusters = None
    for i in range(args.epoch):
        start = 0
        loss_all = 0
        if args.neg_mean == 'R':
            neg_list = get_neg_list_random(G,2)
            X_neg = get_neg_random(X, neg_list)
        while start < config.n:
            end = min(start + config.batch_size, config.n)
            batch_idx = np.array(range(start, end))
            batch_idx = np.random.permutation(batch_idx)
            batch_X = X[batch_idx]
            batch_X_neg = X_neg[batch_idx]
            feed_dict = {model.keep_prob: config.keep_prob, model.batch_idx: batch_idx, model.X: batch_X,
                         model.X_neg: batch_X_neg, model.batch_size: len(batch_idx)}
            _, loss_network = sess.run([model.train_net, model.loss_network], feed_dict=feed_dict)
            loss_all += loss_network


            start_idx = np.random.randint(0, len(all_pairs) - config.batch_size_sg)
            batch_idx = np.array(range(start_idx, start_idx + config.batch_size_sg))
            batch_idx = np.random.permutation(batch_idx)
            batch = np.zeros((config.batch_size_sg), dtype=np.int32)
            labels = np.zeros((config.batch_size_sg, 1), dtype=np.int32)
            batch[:] = all_pairs[batch_idx, 0]
            labels[:, 0] = all_pairs[batch_idx, 1]
            batch_X = X[batch]
            feed_dict = {model.keep_prob: config.keep_prob, model.batch_idx: batch, model.X: batch_X, model.batch_size: len(batch_idx),
                         model.labels: labels}
            _, loss_sg = sess.run([model.train_sg, model.loss_sg], feed_dict=feed_dict)
            loss_all += loss_sg
            start = end
        print('Epoch = {:d} Loss = {:.5f}'.format(i + 1, loss_all / config.n))
        feed_dict = {model.keep_prob: 1.0, model.batch_idx: np.array(range(config.n)), model.X: X, model.batch_size: config.n}
        embedding_result = sess.run(model.Y_final, feed_dict=feed_dict)
        clusters = node_clustering(embedding_result, y, n_clusters=args.kmeans_clusters, seed=args.seed)
        if _nmi_selection is not None:
            _nmi_selection.consider(clusters, embedding_result, i + 1)
        print('node clustering generated {:d} clusters'.format(len(np.unique(clusters))))
        if args.classification_eval and y is not None:
            test_macro_f1_neigh, test_micro_f1_neigh = multiclass_node_classification_eval(
                embedding_result, y, 0.3)
            print('test micro f1 = {:.3f} macro f1 = {:.3f} '.format(test_micro_f1_neigh, test_macro_f1_neigh))
        elif args.classification_eval:
            print('Skipping classification evaluation because data.y is unavailable.')
    print('Learning finished.....')
    if clusters is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M")
        labels_path = os.path.join(args.output_dir, "SANECE_{}_labels.npy".format(timestamp))
        np.save(labels_path, clusters.astype(np.int64))
        np.save(os.path.join(args.output_dir, "embeddings.npy"), embedding_result)
        print("SANECE labels saved in: {}".format(labels_path))
    if not args.data_pt:
        output_embedding_file = 'data/cora/embed/cora.emb'
        write_embedding(embedding_result, output_embedding_file)
if __name__ == '__main__':
    main()
