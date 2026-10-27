import numpy as np
import networkx as nx
import os
import scipy
import torch
from scipy import sparse
from sklearn.preprocessing import normalize

def Pubmed_read_node(data_dir, dataset):
    with open(data_dir + "/{}/Pubmed-Diabetes.NODE.paper.tab".format(dataset), 'rb') as f:
        raw_idx_features_labels = f.read().decode("utf-8")

    raw_idx_features_labels =  raw_idx_features_labels.split('\n')
    feature_name = {}
    for i, f in enumerate(raw_idx_features_labels[1].split('\t')[1:-1]):
        feature_name[f.split(':')[1]] = i
    nodeid = []
    features = []
    labels = []
    for i, s in enumerate(raw_idx_features_labels):
        ss = s.split('\t')
        if ss[0].isdigit():
            nodeid.append(int(ss[0]))
            labels.append(int(ss[1][-1]))
            s_feature = np.zeros(len(feature_name))
            for f in ss[2:-1]:
                ff = f.split('=')
                s_feature[feature_name[ff[0]]] = float(ff[1])
            features.append(s_feature)
        else:
            print(ss[0])
    nodeid = np.array(nodeid)
    features = np.array(features)
    labels = np.array(labels)    
    return nodeid, features, labels

def Pubmed_read_edges(data_dir, dataset):
    with open(data_dir + "/{}/Pubmed-Diabetes.DIRECTED.cites.tab".format(dataset), 'rb') as f:
        raw_adj = f.read().decode("utf-8")
    edges_unordered = []
    for edge in raw_adj.split('\n')[2:-1]:
        s_edge = edge.split('\t')
        e1 = int(s_edge[1].split(':')[-1])
        e2 = int(s_edge[-1].split(':')[-1])
        edges_unordered.append([e1, e2])
    return edges_unordered

def mat_process(data_dir, dataset):
    data_dict = scipy.io.loadmat(data_dir+"/{}/{}.mat".format(dataset, dataset))
    adj = data_dict['Network']
    features = normalize(data_dict['Attributes'],'l2' )
    m, n = features.shape
    dict_NtoC = data_dict['Label']
    lmin = (dict_NtoC.reshape(-1)).min()
    lmax = (dict_NtoC.reshape(-1)).max()
    dict_NtoC = {i:[int(x[0])+(-lmin)] for i, x in enumerate(dict_NtoC)}
    print("Data with node: ", m, "feature:", n, "classes: ", lmax-lmin+1)
    return (adj, features, dict_NtoC)

def _torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')

def _as_numpy(value):
    if hasattr(value, 'is_sparse') and value.is_sparse:
        value = value.to_dense()
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)

def _get_data_attr(data, name):
    if isinstance(data, dict):
        return data.get(name)
    return getattr(data, name, None)

def _unwrap_pyg_data(obj):
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if _get_data_attr(item, 'x') is not None and _get_data_attr(item, 'edge_index') is not None:
                return item
    return obj

def _labels_to_dict(labels):
    labels = _as_numpy(labels)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.reshape(-1)
    if labels.ndim == 1:
        return {i: [int(label)] for i, label in enumerate(labels)}

    dict_NtoC = {}
    unique_values = set(np.unique(labels).tolist())
    is_multihot = unique_values.issubset({0, 1, False, True})
    for i, row in enumerate(labels):
        if is_multihot:
            label = np.flatnonzero(row).astype(int).tolist()
            dict_NtoC[i] = label if label else [int(np.argmax(row))]
        else:
            dict_NtoC[i] = [int(np.argmax(row))]
    return dict_NtoC

def _resolve_data_pt(args):
    data_pt = getattr(args, 'data_pt', None)
    if data_pt:
        if not os.path.isabs(data_pt):
            data_pt = os.path.abspath(data_pt)
        if not os.path.exists(data_pt):
            raise FileNotFoundError("data_pt not found: {}".format(data_pt))
        return data_pt

    dataset_names = []
    for name in [args.dataset, args.dataset.lower(), args.dataset.capitalize(), args.dataset.title()]:
        if name not in dataset_names:
            dataset_names.append(name)

    for dataset in dataset_names:
        for suffix in [('processed', 'data.pt'), ('data.pt',)]:
            path = os.path.join(args.data_dir, dataset, *suffix)
            if os.path.exists(path):
                return path
    return None

def pyg_data_pt_process(args, data_pt):
    data = _unwrap_pyg_data(_torch_load(data_pt))
    x = _get_data_attr(data, 'x')
    y = _get_data_attr(data, 'y')
    edge_index = _get_data_attr(data, 'edge_index')

    if x is None or edge_index is None:
        raise ValueError("Expected data.pt with x and edge_index attributes.")

    features_np = _as_numpy(x).astype('float64')
    edge_index = _as_numpy(edge_index).astype(np.int64)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    num_nodes = features_np.shape[0]
    values = np.ones(edge_index.shape[1], dtype=np.float32)
    adj = sparse.coo_matrix((values, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    if not args.dircted:
        adj = adj.maximum(adj.T).tocsr()

    features = sparse.csr_matrix(features_np)
    dict_NtoC = _labels_to_dict(y) if y is not None else {i: [] for i in range(num_nodes)}
    labels = [label for labels_i in dict_NtoC.values() for label in labels_i]
    print("Loading PyG data.pt: ", data_pt)
    print(
        "Data with node: ", features.shape[0], "feature:", features.shape[1],
        "classes: ", len(set(labels)) if labels else "unavailable",
    )
    return (adj, features, dict_NtoC)

def read_dataset(args):
    data_dir, dataset, data = args.data_dir, args.dataset, args.data
    data_pt = _resolve_data_pt(args)
    if data_pt:
        network_tuple = pyg_data_pt_process(args, data_pt)
        print('Loading Done...')
        return network_tuple

    if args.dataset in ['BlogCatalog', 'Flickr', 'Flickr_SDM']:
        network_tuple = mat_process(data_dir, dataset)
        print('Loading Done...')
        return network_tuple
    if dataset=='Pubmed':
        nodeid, features, labels = Pubmed_read_node(data_dir, dataset)
    elif dataset in ['cora', 'citeseer','cornell', 'texas', 'washington', 'wisconsin']:
        #read index-feature-label
        idx_features_labels = np.genfromtxt(data_dir+"/{}/{}.content".format(dataset, data), dtype=np.dtype(str))
        nodeid = idx_features_labels[:, 0]
        features = idx_features_labels[:, 1:-1].astype('float64')
        print(features.shape)
        labels = idx_features_labels[:, -1]

    if dataset in ['BlogCatalog', 'Flickr', 'Pubmed', 'cora', 'citeseer','cornell', 'texas', 'washington', 'wisconsin']:
        #map label
        classes = set(labels)
        classes_map = {c:i for i,c in enumerate(classes)}
        #remap id
        idstack = []
        IDstr2int = {} 
        newid = 0
        for id in nodeid:
            idstack.append(newid)
            IDstr2int[str(id)] = newid
            newid += 1
            

        nodeid = np.array(idstack)
        node2label = {int(n):classes_map[l] for n, l in zip(nodeid, labels)}
        print("Data with node: ",features.shape[0], "feature:", features.shape[1], "classes: ", len(classes))

        #read edges
        if dataset=='Pubmed':
            edges_unordered = Pubmed_read_edges(data_dir, dataset)
        else:
            edges_unordered = np.genfromtxt("./datasets/{}/{}.cites".format(dataset, data), dtype=str)
        #remove no-feat node
        edges_intID = []
        no_Feat_node = []
        for edge in edges_unordered:
          e1 = edge[0]
          e2 = edge[1]
          if str(e1) in IDstr2int and str(e2) in IDstr2int:
            edges_intID.append([IDstr2int[str(e1)], IDstr2int[str(e2)]])
          else:
            if str(e1) not in IDstr2int and str(e1) not in no_Feat_node:
              no_Feat_node.append(str(e1))
            if str(e2) not in IDstr2int and str(e2) not in no_Feat_node:
              no_Feat_node.append(str(e2))
              
        edges_intID = np.array(edges_intID)
        g = nx.Graph()
        g.add_edges_from(edges_intID)
        #
        error_node = []
        for node_index, features_series in zip(nodeid, features):
            node_index = int(node_index)
          # Haven't yet seen node (not in edgelist) --> add it now
            if not g.has_node(node_index):
                print(node_index, " does not has edge")
                error_node.append(node_index)
            else:
                g.node[node_index]['features'] = features_series

                if node_index in node2label:
                    g.node[node_index]['label'] = [node2label[node_index]]
                else:
                    g.node[node_index]['label'] = []
                    print(node_index, " does not has label")
        print("#Error node: ", len(error_node), '#node after processing: ', features.shape[0]-len(error_node))
        #check connected
        #if not nx.is_connected(g):
        #    print('number_connected_components: ', nx.number_connected_components(g))

        # Get adjacency matrix in sparse format (sorted by g.nodes())
        adj = nx.adjacency_matrix(g) 
        # Get features matrix (also sorted by g.nodes())
        features = np.zeros_like(features) # num nodes, num features

        #
        dict_NtoC ={}
        for i, node in enumerate(g.nodes()):
            features[i,:] = g.node[node]['features']
            dict_NtoC[i] = g.node[node]['label']

        # Save adj, features in pickle file
        features = sparse.csr_matrix(features)
    

        network_tuple = (adj, features, dict_NtoC)
    elif dataset in ['email', 'wiki', 'acm', 'dblp']:
        network_tuple = read_dataset_small(data_dir, dataset)

    print('Loading Done...')
    return network_tuple #adj, features, dict_NtoC

def read_dataset_small(data_dir, dataset):#'email', 'wiki', 'acm', 'dblp'
    data_dir, dataset, data = args.data_dir, args.dataset
    print('reading {} dataset...'.format(dataset))
    sep = '\t' if dataset in ['wiki'] else ' '

    if dataset in ['acm', 'dblp']:
        df = pd.read_csv( data_dir+'/{}/{}.txt'.format(dataset, dataset), header=None, sep=sep)
        df.columns = ['source', 'target']
    else:
        df = pd.read_csv( data_dir+'/{}/{}.txt'.format(dataset, dataset), header=None, sep=sep, names=['source', 'target'])

    graph = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)
    
    if dataset in ['acm', 'dblp']:
        df_label = pd.read_csv(data_dir+'/{}/{}_labels.txt'.format(dataset, dataset), header=None).reset_index()
        df_label.columns = ['node_id', 'label']
    else:
        df_label = pd.read_csv(data_dir+'/{}/{}_labels.txt'.format(dataset, dataset), header=None, sep=sep, names=['node_id', 'label'])

    try:
        if dataset in ['acm', 'dblp']:
            df_feat = pd.read_csv('./datasets/{0}/{1}_features.txt'.format(dataset, dataset), header = None, sep=sep)
            features = sp.csr_matrix(df_feat.values)
        else:
            df_feat = pd.read_csv('./datasets/{0}/{1}_features.txt'.format(dataset, dataset), header = None, sep=sep)
            row = df_feat[0].values
            col = df_feat[1].values
            data = df_feat[2].values
            features = sparse.csr_matrix((data, (row, col)), shape=(row.max()+1, col.max()+1))
        print('find and load features...')
    except:
        features = sparse.eye(adj.shape[0]).tocsr()

    for v in df_label['node_id'].values:
        graph.add_node(v)
    adj = nx.adjacency_matrix(graph)

    le = preprocessing.LabelEncoder()
    le.fit(df_label['label'])
    dict_NtoC = {v[0]:le.transform([v[1]]) for v in df_label.values}
    l_set = set([i[0] for i in dict_NtoC.values()])
    assert max(l_set)-min(l_set)+1 == len(l_set)
    print(adj.shape, features.shape, len(dict_NtoC))
    network_tuple = (adj, features, dict_NtoC)
    
    # save_obj(network_tuple, data_dir+'/{}/adj-feat-label'.format(dataset, dataset))
    print('Loading Done...')
    return network_tuple
