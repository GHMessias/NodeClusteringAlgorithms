import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


def cluster_acc(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must contain the same number of samples")

    _, true_inverse = np.unique(y_true, return_inverse=True)
    _, pred_inverse = np.unique(y_pred, return_inverse=True)
    n_true = int(true_inverse.max()) + 1
    n_pred = int(pred_inverse.max()) + 1

    # Match the common evaluator's rectangular Hungarian assignment. Any
    # unmatched predicted clusters use the same fallback label as the runner.
    contingency = np.zeros((n_true, n_pred), dtype=np.int64)
    np.add.at(contingency, (true_inverse, pred_inverse), 1)
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)

    mapping = np.zeros(n_pred, dtype=np.int64)
    mapping[col_ind] = row_ind
    new_predict = mapping[pred_inverse]

    acc = metrics.accuracy_score(true_inverse, new_predict)
    f1_macro = metrics.f1_score(
        true_inverse, new_predict, average='macro', zero_division=0
    )
    return acc, f1_macro


def eva(y_true, y_pred, epoch=0, ACC=None, NMI=None, ARI=None, F1=None, verbosity=5):
    if y_true is None:
        acc = nmi = ari = f1 = float('nan')
    else:
        acc, f1 = cluster_acc(y_true, y_pred)
        nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
        ari = ari_score(y_true, y_pred)
    #if verbosity > 3: print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari), ', f1 {:.4f}'.format(f1))
    if verbosity > 3:
        if y_true is None:
            print(f'{epoch}: ground-truth labels unavailable; skipping external metrics')
        else:
            print(f'{epoch} acc:{acc:.4f}  nmi:{nmi:.4f}  ari:{ari:.4f}  f1:{f1:.4f}')

    if ACC is not None: ACC.append(acc)
    if NMI is not None: NMI.append(nmi)
    if ARI is not None: ARI.append(ari)
    if F1 is not None: F1.append(f1)
    return acc,nmi,ari,f1

def display(data_list):
    return f'{np.max(data_list):7.4f}, {np.min(data_list):7.4f}, {np.mean(data_list):7.4f}, {np.std(data_list):7.4f}'
