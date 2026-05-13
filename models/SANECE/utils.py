import networkx as nx
import numpy as np
import node2vec
from sklearn.metrics.pairwise import cosine_distances
import random
from math import log

def read_label(inputFileName):
    f = open(inputFileName, "r")
    labels = []
    for line in f.readlines():
        line = line.strip("\n").split(" ")
        labels.append(int(line[1]))
    f.close()
    return labels


def getRandomWalkWholePairs(G,directed,p,q,num_walks,walk_length,window_size):
    def generate_graph_context_all_pairs(path, window_size):
        all_pairs = []
        for k in range(len(path)):
            for i in range(len(path[k])):
                for j in range(i - window_size, i + window_size + 1):
                    if i == j or j < 0 or j >= len(path[k]):
                        continue
                    else:
                        all_pairs.append([path[k][i], path[k][j]])
        return np.array(all_pairs, dtype=np.int32)

    print('generating random walks')
    G2 = node2vec.Graph(G, directed, p, q)
    G2.preprocess_transition_probs()
    walks = G2.simulate_walks(num_walks, walk_length)
    whole_pairs = generate_graph_context_all_pairs(walks, window_size)
    return whole_pairs

def write_embedding(embedding_result,outputFileName):
    f = open(outputFileName,'w')
    N,dims = embedding_result.shape
    for i in range(N):
        s = ''
        for j in range(dims):
            if j == 0:
                s = str(i) + ',' + str(embedding_result[i][j])
            else:
                s = s + ',' + str(embedding_result[i][j])
        f.writelines(s+'\n')
    f.close()

def get_neg_aa(X, aa_list, dataset):
    m = X.shape[1]
    neg_mtx = np.zeros((len(aa_list), m))
    for i in range(len(aa_list)):
        neg_mtx[i] = X[int(aa_list[i])]
    return neg_mtx

def get_neg_list_aa(G, k):
    def _apply_prediction(G, func, ebunch=None):

        if ebunch is None:
            ebunch = nx.non_edges(G)
        return ((u, v, func(u, v)) for u, v in ebunch)

    def adamic_adar_index(G, ebunch=None):
        def predict(u, v):
            return sum(1 / log((G.degree(w) + 1e-5)) for w in nx.common_neighbors(G, u, v))

        return _apply_prediction(G, predict, ebunch)
    def AA_similarity_numpy(G, node, target):
        evaluate_list = []
        simi_list = []
        for j in target:
            evaluate_list.append((node, j))
        preds = adamic_adar_index(G, evaluate_list)
        for u, v, p in preds:
            simi_list.append(p)
        return simi_list
    neg_list = []
    for node in G.nodes():
        a = nx.single_source_shortest_path_length(G, node, cutoff=k)
        neighbors = list(a.keys())
        simi_list = list(AA_similarity_numpy(G, node, neighbors))
        min_index = simi_list.index(min(simi_list))
        while neighbors[min_index] in neg_list:
            if len(neighbors) == 1:
                break
            else:
                neighbors.remove(neighbors[min_index])
                simi_list = list(AA_similarity_numpy(G, node, neighbors))
                min_index = simi_list.index(min(simi_list))
        neg_list.append(neighbors[min_index])

    return neg_list

def get_neg_list_random(G, k):
    neg_list = []
    for node in G.nodes():
        a = nx.single_source_shortest_path_length(G, node, cutoff=k)
        neighbors = list(set(np.arange(0, len(G.nodes()))) - set(list(a.keys())))
        neg_list.append(random.choice(neighbors))
    return neg_list

def get_neg_random(X, neg_list):
    neg_mtx = np.zeros(X.shape)
    for i in range(len(neg_list)):
        neg_mtx[i] = X[int(neg_list[i])]
    return neg_mtx

def get_neg_list_attr(G, X, k):
    neg_list = []
    for node in G.nodes():
        a = nx.single_source_shortest_path_length(G, node, cutoff=k)
        neighbors = list(a.keys())
        dis = list(cosine_distances(X[node].reshape(1, -1), X[neighbors])[0])
        neg_list.append(neighbors[dis.index(np.max(dis))])
    return neg_list

def get_neg_attr(X, attr_list):
    neg_mtx = np.zeros(X.shape)
    for i in range(len(attr_list)):
        neg_mtx[i] = X[int(attr_list[i])]
    return neg_mtx
