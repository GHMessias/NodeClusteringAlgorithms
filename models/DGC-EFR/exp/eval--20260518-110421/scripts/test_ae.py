import numpy as np
import h5py
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva
from utils import load_data
# from GNN import GCN, GAT, SpGAT
#torch.cuda.set_device(3)
from visualize import visualize_data_tsne


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3,  n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z,  v=1):
        super(AE, self).__init__()

        # autoencoder for intra information
        self.enc_1 = Linear(n_input, n_enc_1)  # 3703 -> 1024
        self.enc_2 = Linear(n_enc_1, n_enc_2)  # 1024 -> 256
        self.enc_3 = Linear(n_enc_2, n_enc_3)  # 256  -> 16
        self.z_layer = Linear(n_enc_3, n_z)    # 16  -> 10

        self.dec_1 = Linear(n_z, n_dec_1)      # 10  -> 16
        self.dec_2 = Linear(n_dec_1, n_dec_2)  # 16 -> 256
        self.dec_3 = Linear(n_dec_2, n_dec_3)  # 256 -> 1024
        self.x_bar_layer = Linear(n_dec_3, n_input)  # 1024 -> 3703
        # GCN for inter information

        # degree
        self.v = v

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))        # 3703 -> 1024
        enc_h2 = F.relu(self.enc_2(enc_h1))   # 1024 -> 256
        enc_h3 = F.relu(self.enc_3(enc_h2))   # 256  -> 16
        z = self.z_layer(enc_h3)              # 16  -> 10

        dec_h1 = F.relu(self.dec_1(z))        # 10  -> 16
        dec_h2 = F.relu(self.dec_2(dec_h1))   # 16 -> 256
        dec_h3 = F.relu(self.dec_3(dec_h2))   # 256 -> 1024
        x_bar = self.x_bar_layer(dec_h3)      # 1024 -> 3703

        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test_ae(model, dataset, args, y):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    # x = torch.Tensor(dataset.x).cuda()

    with torch.no_grad():
        x = torch.Tensor(dataset.x).cuda().float()
        x_bar, z = model(x)
        loss = F.mse_loss(x_bar, x)
        print('loss: {}'.format(loss))
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
        eva(y, kmeans.labels_, 0)
        visualize_data_tsne(z.cpu().detach().numpy(), y, args.n_clusters,
                            'exp/'+args.name+'_tsne-ae.pdf')
        # torch.save(model.state_dict(), 'data/{}_ae.pkl'.format(args.name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='acm')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=128, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'data/{}_ae.pkl'.format(args.name)

    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561

    if args.name == 'reut':
        args.lr = 1e-4
        args.n_clusters = 4
        args.n_input = 2000

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334

    if args.name == 'cora':
        args.k = None
        args.n_clusters = 7
        args.n_input = 1433

    if args.name == 'cite':
        # args.lr = 1e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703

    x_path = 'data/{}.txt'.format(args.name)
    y_path = 'data/{}_label.txt'.format(args.name)
    x = np.loadtxt(x_path, dtype=float)
    y = np.loadtxt(y_path, dtype=int)

    model = AE(
            n_enc_1=1024,
            n_enc_2=256,
            n_enc_3=16,
            n_dec_1=16,
            n_dec_2=256,
            n_dec_3=1024,
            n_input=args.n_input,
            n_z=10,).cuda()
    model.load_state_dict(torch.load(args.pretrain_path))
    dataset = LoadDataset(x)
    test_ae(model, dataset,args, y)