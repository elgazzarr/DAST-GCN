import pickle
import numpy as np
import os
import torch
import sklearn.metrics as results
import scipy.sparse as sp


def sparsing(x, th):
    x = np.abs(x)
    x_flattened = x.flatten()
    l = x_flattened.shape[0] // th
    x_sorted = np.sort(x_flattened)[::-1]
    ths = x_sorted[l]
    x[x >= ths] = 1
    x[x < ths] = 0
    # print('N of edges={}'.format(np.count_nonzero(x)))
    return x, ths

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class Arguments:
    n_samples = 2000
    device = 'cuda:0'
    in_dim = 3
    n_filters = 8
    batch_size = 32
    lr = 0.001
    dropout = 0.1
    weight_decay = 0.00001
    epochs = 150
    save = '/data/agelgazzar/projects/DAST-GCN/models/'
    layers = 5
    kernel = 3
    pool = False
    res = True








def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data







def inference(engine,loader,device='cuda:0'):
    outputs = []
    labels = []

    for iter, (x, y) in enumerate(loader):
        testx = x.float().to(device)
        testx = testx.transpose(1,3)

        with torch.no_grad():
            y = torch.argmax(y,1)
            preds = engine.model(testx)
            preds = torch.argmax(preds,1).cpu().numpy()

        labels.append(y)
        outputs.append(preds)

    preds = np.concatenate(outputs)
    labels = np.concatenate(labels)

    return {'Auc': round(results.roc_auc_score(labels,preds),3),
        "Acc": round(results.accuracy_score(labels, preds),3),
        "Prec": round(results.precision_score(labels, preds),3),
        "Recall": round(results.recall_score(labels, preds),3)}

def inference_regress(engine,loader,device='cuda:0'):
    outputs = []
    labels = []

    for iter, (x, y) in enumerate(loader):
        testx = x.float().to(device)
        testx = testx.transpose(1,3)

        with torch.no_grad():
            preds = engine.model(testx).cpu().numpy()

        labels.append(y)
        outputs.append(preds)

    preds = np.concatenate(outputs)
    print(preds)
    labels = np.concatenate(labels)

    return {'MSE': round(results.mean_squared_error(labels,preds),3),
        "MAE": round(results.mean_absolute_error(labels, preds),3),
        "R2": round(results.r2_score(labels, preds),3)}
