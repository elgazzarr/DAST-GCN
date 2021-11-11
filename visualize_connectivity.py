import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from engine import trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from nilearn import datasets
from nilearn import plotting
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--df',type=str,default='./csvfiles/ukbb_10k.csv',help='')
parser.add_argument('--N',type=int,default=5000)
parser.add_argument('--label',type=str,default='Sex')
parser.add_argument('--cv',default=True,help='5fold cross validation')
parser.add_argument('--dynamic',default=True,help='whether only adaptive adj')
parser.add_argument('--gcn_bool',default=True,help='whether to add graph convolution layer')
parser.add_argument('--aptonly',default=True,help='whether only adaptive adj')
parser.add_argument('--n_blocks',type=int,default=3,help='number of st-gcn blocks')
parser.add_argument('--nhid',type=int,default=10,help='number of filtes in hidden layers')
parser.add_argument('--kernel',type=int,default=3,help='number of filtes in hidden layers')
parser.add_argument('--in_dim',type=int,default=3,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=116,help='number of nodes')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.5,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=150 ,help='')
parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./models/gender_dynamic_3/',help='save path')
args = parser.parse_args()

def visualize(fold=0,th=0.5):

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # load data

    device = torch.device(args.device)

    engine = trainer(dynamic=args.dynamic, in_dim=args.in_dim, kernel=args.kernel, num_nodes=args.num_nodes, filters=args.nhid , dropout=args.dropout, lrate=args.lr, wdecay=args.weight_decay, device=device, supports=None, blocks=args.n_blocks)
    model = engine.model
    model.load_state_dict(
        torch.load(args.save + "best_fold-" + str(fold) + ".pth"))

    if args.dynamic:
        for i in range(len(model.adjs_s)):


            X = F.softmax(F.relu(torch.mm(model.adjs_s[i], model.adjs_t[i])), dim=1).detach().cpu().numpy()
            Xs = (X - np.min(X)) / np.ptp(X)
            thr_ind = Xs < th
            Xs[thr_ind] = 0
            inds = np.where(X > th)
            labels = plot_matrix(X, i, fold)
            get_regions(X, inds, labels, fold)
            print('plotted adj-f{}_{}'.format(fold,i+1))
        print('Done.')
    else:
        Xs = F.softmax(F.relu(torch.mm(model.adjs_s, model.adjs_s.transpose(1,0))),dim=1).detach().cpu().numpy()
        #Xs = (X - np.min(X)) / np.ptp(X)
        thr_ind = Xs < th
        Xs[thr_ind] = 0
        inds = np.where(Xs > th)
        labels = plot_matrix(Xs, -1, fold)

        get_regions(Xs, inds,labels,fold)

        print('plotted adj-f{}_{}'.format(fold, 0))

def get_regions(x, inds,regions,fold):

    names = []
    values = []
    for i in range(inds[0].shape[0]):
        r_1 = inds[0][i]
        r_2 = inds[1][i]
        r1 = regions[r_1]
        r2 = regions[r_2]
        value = x[r_1,r_2]
        conn = '{} and {}'.format(r1,r2,value)
        if value not in values:
            names.append(conn)
            values.append(value)

    s = np.argsort(values)
    s = s[::-1]
    values = np.sort(values)
    values = values[::-1]
    names = [names[j] for j in s]
    fname = args.save + '/connections-f{}.txt'.format(fold)
    with open(fname, 'a') as f:
        for n,v in zip(names,values):
            print(n,', value = {:1f}'.format(v),file=f)
        print('-'*30,file=f)


def plot_matrix(adj,i,fold):

    dataset = datasets.fetch_atlas_aal('SPM12')
    labels = dataset.labels
    #df = pd.read_csv('AAL_116.csv')
    #coords = np.array([df.X, df.Y, df.Z]).transpose()
    coords = plotting.find_parcellation_cut_coords(dataset.maps)
    display = plotting.plot_matrix(adj, figure=(12, 12), labels=labels
                         )
    display.figure.savefig(args.save+'adj-f{}_{}'.format(fold,i+1))

    graph = plotting.plot_connectome(adj, coords, node_size=2)
    graph.savefig(args.save + 'graph-f{}+{}'.format(fold,i+1))
    return labels


if __name__ == "__main__":
    visualize()
