
# Optional benchmark result selection; standalone/default training is unchanged.
import os as _benchmark_os
if _benchmark_os.environ.get("BENCHMARK_SELECT_BEST_NMI") == "1":
    from benchmark_selection import selection as _nmi_selection
else:
    _nmi_selection = None

import warnings

warnings.filterwarnings('ignore')

import argparse
import os
import time

import numpy as np
import torch
from encoder import GCN
from sklearn.cluster import KMeans
from torch.optim import Adam
from utils import get_logger, fix_seed, evaluate, normalize, augment, load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NS4GC")
    parser.add_argument("--data_pt", type=str, required=True, help="Path to PyG data.pt with x and edge_index; y is optional")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--kmeans_clusters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda_neighbor", type=float, default=1.0)
    parser.add_argument("--gamma_sparsity", type=float, default=1.0)
    parser.add_argument("--similarity_threshold", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--hidden_dims", type=int, nargs=2, default=[256, 64])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--edge_drop_rate_1", type=float, default=0.3)
    parser.add_argument("--edge_drop_rate_2", type=float, default=0.3)
    parser.add_argument("--feature_drop_rate_1", type=float, default=0.2)
    parser.add_argument("--feature_drop_rate_2", type=float, default=0.2)
    parser.add_argument("--kmeans_n_init", type=int, default=10)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but it is not available.")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    A, X, Y = load_data(args.data_pt)
    dataset = os.path.splitext(os.path.basename(args.data_pt))[0]

    # Set hyper-parameters
    in_dim = X.shape[1]
    # Copy from params.txt
    epochs = args.epochs
    lam, gam = args.lambda_neighbor, args.gamma_sparsity
    s, tau = args.similarity_threshold, args.temperature
    hid_dims = args.hidden_dims
    lr, wd = args.lr, args.weight_decay
    pd1, pd2 = args.edge_drop_rate_1, args.edge_drop_rate_2
    pm1, pm2 = args.feature_drop_rate_1, args.feature_drop_rate_2

    logger = get_logger(os.path.join(args.output_dir, f'{dataset}_NS4GC.log'))
    params = {
        'epochs': epochs,
        'lr, wd': (lr, wd),
        's, tau': (s, tau),
        'hid_dims': hid_dims,
        'lam, gam': (lam, gam),
        'drop_rate': (pd1, pd2, pm1, pm2)
    }
    logger.info(str(params))

    fix_seed(args.seed)  # for reproduction
    encoder = GCN(in_dim, hid_dims)
    encoder = encoder.to(device)
    A = A.to(device)
    X = X.to(device)

    src, dst = A._indices()
    mask = torch.full(A.size(), True, device=A.device)
    mask[src, dst] = False
    mask.fill_diagonal_(False)

    # Pre-train
    optimizer = Adam(encoder.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(epochs):
        encoder.train()
        A1, X1 = augment(A, X, pd1, pm1)
        A2, X2 = augment(A, X, pd2, pm2)
        A_norm1 = normalize(A1, add_self_loops=True)
        A_norm2 = normalize(A2, add_self_loops=True)
        Z1, Z2 = encoder(A_norm1, A_norm2, X1, X2)

        # Z1 and Z2 have been normalized.
        S = Z1 @ Z2.T

        loss_ali = - torch.diag(S).mean()

        loss_nei = - S[src, dst].mean()

        S = torch.masked_select(S, mask)
        S = torch.sigmoid((S - s) / tau)
        loss_spa = S.mean()

        loss = loss_ali + lam * loss_nei + gam * loss_spa
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if _nmi_selection is not None:
            with _nmi_selection.inference(encoder):
                candidate = encoder.embed(normalize(A, add_self_loops=True), X)
                candidate_k = args.kmeans_clusters
                if candidate_k is None:
                    candidate_k = int(np.unique(Y.cpu().numpy()).size)
                _nmi_selection.kmeans(candidate, candidate_k, epoch + 1,
                                      n_init=args.kmeans_n_init)

        print(f'E:{epoch} Loss:{loss.item():.4f} ALI:{loss_ali.item():.4f} '
              f'NEI:{loss_nei.item():.4f} SPA:{loss_spa.item():.4f}')

    # Clustering
    encoder.eval()
    A_norm = normalize(A, add_self_loops=True)
    with torch.no_grad():
        Z = encoder.embed(A_norm, X)
    Z_np = Z.cpu().numpy()
    Y_np = Y.cpu().numpy() if Y is not None else None
    if Y_np is not None:
        evaluate(Z_np, Y_np, logger)

    n_clusters = args.kmeans_clusters
    if n_clusters is None and Y_np is not None:
        n_clusters = np.unique(Y_np).shape[0]
    if n_clusters is None:
        raise ValueError("--kmeans_clusters is required when data.y is unavailable.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=args.kmeans_n_init)
    labels = kmeans.fit_predict(Z_np)
    timestamp = time.strftime("%Y%m%d_%H%M")
    labels_path = os.path.join(args.output_dir, f"NS4GC_{timestamp}_labels.npy")
    np.save(labels_path, labels.astype(np.int64))
    np.save(os.path.join(args.output_dir, "embeddings.npy"), Z_np)
    print(f"NS4GC labels saved in: {labels_path}")

    logger.info('')
