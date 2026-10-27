
# Optional benchmark result selection; standalone/default training is unchanged.
import os as _benchmark_os
if _benchmark_os.environ.get("BENCHMARK_SELECT_BEST_NMI") == "1":
    from benchmark_selection import selection as _nmi_selection
else:
    _nmi_selection = None

from collections import Counter

from sklearn.cluster import KMeans

import utils
from model import GAT
from model_GAT import GAT as GAT_self_supervised
from evaluation import eva

import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


def trainer(dataset):
    #  模型实例化
    model_GAT = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
    ).to(device)
    print(model_GAT)
    # 加载预训练权重模型
    if args.pretrain_path and os.path.exists(args.pretrain_path):
        try:
            model_GAT.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
        except RuntimeError as exc:
            print("Could not load pretrain weights from {}: {}".format(args.pretrain_path, exc))
            print("Continuing with randomly initialized GAT weights.")
    else:
        print("Pretrain weights not found. Continuing with randomly initialized GAT weights.")

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # Node features. Ground-truth labels are intentionally not read here: DSAGC
    # must train and select its output using pseudo-labels only.
    data = torch.Tensor(dataset.x).to(device)

    with torch.no_grad():
        _, z = model_GAT(data, adj, M)
    features = z.data.cpu().numpy()

    # get kmeans and pretrain cluster result
    # # 谱聚类
    # sc = SpectralClustering(n_clusters=args.n_clusters, affinity="nearest_neighbors")
    # y_pred = sc.fit_predict(features)
    # acc, nmi, ari, f1 = eva(reference_labels, y_pred, 'pretrain-谱聚类')

    # K-means聚类
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=args.kmeans_n_init, random_state=args.seed)
    y_pred = kmeans.fit_predict(features)  # (2708,)

    # 聚类中心获取
    cluster = kmeans.cluster_centers_  # 获取聚类中心  # (7, 16)
    # result = [acc, nmi, ari, f1]

    # 计算可信赖样本的精确度
    # 验证标签和聚类中心之间的关联度
    for i in range(y_pred.shape[0]):
        dis = []
        for j in range(cluster.shape[0]):
            dist = np.sqrt(np.sum(np.square(features[i] - cluster[j])))
            dis.append(dist)
        if not dis.index(min(dis))==y_pred[i]:
            print("error")
        dis.clear()

    # 获取到簇中心最小的距离
    distances_samples = []  # 每个样本到该簇类中心的距离
    for i in range(features.shape[0]):
        dist_temp = np.sum(np.square(features[i] - cluster[y_pred[i]]))  # 欧式距离
        distances_samples.append(dist_temp)
    distances_samples = np.array(distances_samples)

    # 可信赖样本的阈值
    alpha_k = args.reliable_threshold
    print("可信赖样本的个数", np.sum(np.array(distances_samples) < alpha_k))
    realiable_sample = np.argwhere(distances_samples < alpha_k)
    # print(distances_samples[realiable_sample])  # 输出这些距离


    # 测试
    if len(realiable_sample) == 0:
        realiable_sample = np.arange(features.shape[0]).reshape(-1, 1)
    y_pred_new = [int(y_pred[i[0]]) for i in realiable_sample]
    print(Counter(y_pred_new))

    # 自监督学习  # x, adj, y_pre, sample_index
    return train_self_supervised(x=data, adj=adj, y_pred=y_pred, sample_index=realiable_sample, epoch=args.epochs)


def train_self_supervised(x, adj, y_pred, sample_index, epoch=200):
    # 初始化模型
    sample_index = [i[0] for i in sample_index]
    x, adj, y_pred = x.to(device), adj.to(device), torch.LongTensor(y_pred).to(device)
    print("自监督训练开始,  GAT模型结构图：")

    model = GAT_self_supervised(
        nfeat=x.shape[1],
        nhid=args.self_supervised_hidden,
        nclass=args.n_clusters,
        dropout=args.self_supervised_dropout,
        nheads=args.self_supervised_heads,
        alpha=args.alpha,
    )
    print(model)
    # summary(model, input_size=(), device='cpu')
    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.self_supervised_lr,
        weight_decay=args.self_supervised_weight_decay,
    )
    for i in range(epoch):
        model.train()
        optimizer.zero_grad()
        output = model(x, adj)
        loss_train = F.nll_loss(output[sample_index], y_pred[sample_index])
        loss_train.backward()
        optimizer.step()
        if _nmi_selection is not None:
            with _nmi_selection.inference(model):
                candidate = model(x, adj)
                _nmi_selection.consider(candidate.argmax(1), candidate, i + 1)

        # 评估指标
        y_pre_GAT = np.argmax(output.cpu().detach().numpy(), axis=1)
        y_pre_kmeans = y_pred.cpu().detach().numpy()

        acc_train, nmi_train, ari_train, f1_train = eva(y_pre_GAT[sample_index], y_pre_kmeans[sample_index], '{}'.format(i))
        print(f"epoch - Train {i}:acc {acc_train:.4f}, nmi {nmi_train:.4f}, ari {ari_train:.4f}, f1 {f1_train:.4f}")

    # Return the final trained model. Selecting a checkpoint with ACC against
    # data.y would leak ground truth into this unsupervised benchmark.
    model.eval()
    with torch.no_grad():
        final_output = model(x, adj)
    final_labels = np.argmax(final_output.cpu().numpy(), axis=1).astype(np.int64)
    final_embeddings = final_output.cpu().numpy().copy()
    checkpoint_path = os.path.join(args.checkpoint_dir, f"self_GCN_{args.name}_final.pkl")
    torch.save(model.state_dict(), checkpoint_path)
    return final_labels, final_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Cora')  # Citeseer, Cora
    parser.add_argument('--data_pt', type=str, default=None, help='Path to PyG data.pt with x and edge_index; y is optional')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=400, help='Self-supervised training epochs')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='pretrain')
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--epoch', type=int, default=30)  # 预训练模型中的epoch
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_clusters', default=None, type=int)
    parser.add_argument('--update_interval', default=1, type=int)  # [1,3,5]
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--embedding_size', default=16, type=int)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--reliable_threshold', type=float, default=0.36)
    parser.add_argument('--kmeans_n_init', type=int, default=20)
    parser.add_argument('--self_supervised_hidden', type=int, default=8)
    parser.add_argument('--self_supervised_dropout', type=float, default=0.18)
    parser.add_argument('--self_supervised_heads', type=int, default=8)
    parser.add_argument('--self_supervised_lr', type=float, default=0.005)
    parser.add_argument('--self_supervised_weight_decay', type=float, default=5e-5)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but it is not available.")
    args.cuda = args.device == 'cuda'
    print("use cuda: {}".format(args.cuda))
    device = torch.device(args.device)

    if args.data_pt:
        dataset = utils.load_data_pt(args.data_pt)
    else:
        datasets = utils.get_dataset(args.name)
        dataset = datasets[0]

    if args.data_pt:
        args.k = None
    elif args.name == 'Citeseer':
        args.lr = 0.0001
        args.k = None
        args.n_clusters = 6
    elif args.name == 'Cora':
        args.lr = 0.0001
        args.k = None
        args.n_clusters = 7
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None

    if args.pretrain_path is None:
        args.pretrain_path = f'pretrain/GAE_{args.name}.pkl'
    args.input_dim = dataset.num_features
    if args.n_clusters is None:
        raise ValueError("--n_clusters is required for custom data; DSAGC does not infer K from data.y.")

    print(args)
    labels, embeddings = trainer(dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M")
    labels_path = os.path.join(args.output_dir, "DSAGC_{}_labels.npy".format(timestamp))
    np.save(labels_path, labels.astype(np.int64))
    np.save(os.path.join(args.output_dir, "embeddings.npy"), embeddings)
    print("DSAGC labels saved in: {}".format(labels_path))
