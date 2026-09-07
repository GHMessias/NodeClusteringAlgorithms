import numpy as np
import torch
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva



# the basic autoencoder
class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3,  n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z,  v=1):
        super(AE, self).__init__()

        # encoder configuration
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)
        # decoder configuration
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

        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data.astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
            if get_data_attr(item, 'x') is not None:
                return item
    return obj

def load_data_pt(path):
    data = unwrap_data_pt(torch_load(path))
    x = get_data_attr(data, 'x')
    y = get_data_attr(data, 'y')
    if x is None or y is None:
        raise ValueError("Expected data.pt with x and y attributes.")

    x = as_numpy(x).astype(np.float32)
    y = as_numpy(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    elif y.ndim > 1:
        y = np.argmax(y, axis=1)
    return x, y.astype(int)

# pre-train the autoencoder model
def pretrain_ae(model, dataset, args, y):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    for epoch in range(args.epochs):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)

            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            x = torch.Tensor(dataset.x).to(device).float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
            eva(y, kmeans.labels_, epoch)

    output_dir = os.path.dirname(args.preae_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), args.preae_path)
    print("Saved pre-trained AE: {}".format(args.preae_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='acm')
    parser.add_argument('--data_pt', type=str, default=None, help='Optional PyG data.pt path with x and y')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=None, type=int)
    parser.add_argument('--n_input', default=None, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--preae_path', type=str, default=None)
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.name == 'acm' and not args.data_pt:
        args.k = None
        if args.n_clusters is None:
            args.n_clusters = 3

    if args.name == 'dblp' and not args.data_pt:
        args.k = None
        if args.n_clusters is None:
            args.n_clusters = 4

    if args.name == 'cora' and not args.data_pt:
        args.k = None
        if args.n_clusters is None:
            args.n_clusters = 7

    if args.name == 'cite' and not args.data_pt:
        args.k = None
        if args.n_clusters is None:
            args.n_clusters = 6

    if args.data_pt:
        x, y = load_data_pt(args.data_pt)
    else:
        x_path = 'data/{}.txt'.format(args.name)
        y_path = 'data/{}_label.txt'.format(args.name)
        x = np.loadtxt(x_path, dtype=float).astype(np.float32)
        y = np.loadtxt(y_path, dtype=int)

    if args.n_input is None:
        args.n_input = x.shape[1]
    if args.n_clusters is None:
        args.n_clusters = len(set(np.asarray(y).reshape(-1).tolist()))
    if args.preae_path is None:
        args.preae_path = 'data/preae_{}.pkl'.format(args.name)

    model = AE(
            n_enc_1=1024,
            n_enc_2=256,
            n_enc_3=16,
            n_dec_1=16,
            n_dec_2=256,
            n_dec_3=1024,
            n_input=args.n_input,
            n_z=args.n_z,).to(device)

    dataset = LoadDataset(x)
    pretrain_ae(model, dataset,args, y)
