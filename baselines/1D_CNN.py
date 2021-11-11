import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
import networkx as nx
#import dgl
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
#from dgl.data import MiniGCDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn.metrics as results
import copy
from torch.autograd import Variable
from baselines.data import UkbbData

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.cuda.set_device(1)
class Temporal_Conv(nn.Module):
    """Update the node feature tv with applying Temporal Convolutions to get tv+1."""

    def __init__(self, in_channels, filters, k, d, activation, causal=False):
        super(Temporal_Conv, self).__init__()
        self.causal = causal

        if causal:
            p = (k - 1) * d
        else:
            p = 0

        self.conv1 = nn.Conv1d(in_channels, filters, k, dilation=d, padding=p)
        self.activation = activation
        self.bn = nn.BatchNorm1d(filters)
        self.pool = nn.MaxPool1d(3)

    def forward(self, x):
        h = self.conv1(x)
        if self.causal:
            h = h[:, :, :-self.conv1.padding[0]]
        h = self.bn(h)
        h = self.activation(h)
        h = self.pool(h)

        return h


class Model_1DCNN(nn.Module):
    def __init__(self, nrois, filters=[256, 64, 128], dilations=[1, 1, 1], kernels=[3, 3, 3]):
        super(Model_1DCNN, self).__init__()

        self.layer0 = Temporal_Conv(nrois, filters[0], kernels[0], dilations[0], F.relu)
        self.layer1 = Temporal_Conv(filters[0], filters[1], kernels[1], dilations[1], F.relu)
        self.layer2 = Temporal_Conv(filters[1], filters[2], kernels[2], dilations[2], F.relu)

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=0.5)
        self.classify = nn.Linear(filters[1], 2)

    def forward(self, x):
        verbose = False
        x = x.float().to('cuda:1')
        if verbose:
            print('input shape: ', x.shape)

        h0 = self.layer0(x)  # B,t,C

        h1 = self.layer1(h0)

        #h2 = self.layer2(h1)

        h = self.temporal_pool(h1)
        if verbose:
            print('average pooling shape: ', h.shape)
        h = torch.squeeze(h)

        h = self.drop(h)

        hg = self.classify(h)
        if verbose:
            print('mlp output shape: ', hg.shape)

        hg = nn.functional.softmax(hg, dim=1)
        if verbose:
            print('final output shape: ', hg.shape)
            print('--------------------')

        return hg


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    #if epoch % lr_decay_epoch == 0:
        #print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def train_model(train_loader, val_loader, nrois, epochs):
    # Create training and testing set.

    val_loss = []
    best_val_loss = None
    best_model = None

    model = Model_1DCNN(nrois).cuda()
    loss_func = nn.CrossEntropyLoss().to('cuda:1')
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    es = 8
    es_counter = 0
    print('Training....')
    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for iter, (tc, label) in enumerate(train_loader):
            # tc = tc.to('cuda:0')
            label = Variable(label).type(torch.cuda.LongTensor).to('cuda:1')
            prediction = model(tc).to('cuda:1')
            loss = loss_func(prediction, torch.max(label, 1)[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss = epoch_loss / iter
        optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=lr, lr_decay_epoch=40)

        epoch_val_loss = validate_model(model, val_loader)
        val_loss.append(epoch_val_loss)

        if not best_val_loss or epoch_val_loss < best_val_loss:
            #print('Saving best model ...')
            best_model = copy.deepcopy(model)
            best_val_loss = epoch_val_loss
            es_counter = 0
        if es_counter > es:
            break
        if epoch > 5:
            es_counter += 1

        #print('Epoch {}, train_loss {:.3f}, val_loss {:.3f}'.format(epoch, epoch_loss, epoch_val_loss))
        epoch_losses.append(epoch_loss / (iter + 1))
    #print('Done.')
    return best_model


def validate_model(model, val_loader):
    model.eval()
    val_loss = 0
    loss_func = nn.CrossEntropyLoss().cuda()

    for iter, (tc, label) in enumerate(val_loader):
        label = Variable(label).type(torch.cuda.LongTensor)
        prediction = model(tc).cuda()
        loss = loss_func(prediction.cuda(), torch.max(label, 1)[1].cuda())
        val_loss += loss.detach().item()

    return val_loss / iter


def test_model(model, test_loader):
    model.eval()
    labels = np.empty([], dtype=int)
    predictions = np.empty([], dtype=int)
    print('Testing...')
    for iter, (tc, label) in enumerate(test_loader):
        # tc = tc.to('cuda:0')
        label = Variable(label).type(torch.cuda.LongTensor)
        prediction = model(tc).cuda()

        labels = np.append(labels, torch.argmax(label, 1).cpu().numpy())
        predictions = np.append(predictions, torch.argmax(prediction, 1).cpu().numpy())

    y_test=labels[1:]
    y_pred = predictions[1:]
    accuracy1 = results.balanced_accuracy_score(y_test, y_pred)
    sensitivity1 = results.precision_score(y_test, y_pred)
    specificity1 = results.recall_score(y_test, y_pred)

    return round(accuracy1, 3), round(sensitivity1, 3), round(specificity1, 3)


from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold


def run_kfold(df_path, atlas, epochs=100, bs=16):
    print('Preparing Data ...')
    df = pd.read_csv(df_path)
    df['Sex'].replace({'Male':0, 'Female':1},inplace=True)

    dataset = UkbbData(data_info_file=df, atlas_name=atlas)
    nrois = dataset.nrois

    kf = KFold(n_splits=5)
    kf.get_n_splits(dataset)
    accs = []
    senss = []
    specs = []
    k = 1

    for train_index, test_index in kf.split(dataset):

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(test_index)
        train_loader = DataLoader(dataset, batch_size=bs, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=bs, sampler=valid_sampler)
        test_loader = DataLoader(dataset, batch_size=bs, sampler=valid_sampler)
        print('Training Fold : {}'.format(k))
        best_model = train_model(train_loader, val_loader, nrois, epochs)
        acc, sens, spec = test_model(best_model, test_loader)
        print("Test Accuracy for fold {} = {}".format(k, acc))
        print("Test sens for fold {} = {}".format(k, sens))
        print("Test spec for fold {} = {}".format(k, spec))
        accs.append(acc)
        senss.append(sens)
        specs.append(spec)
        k += 1
    print('---------------------------------')
    print("5 fold Test Accuracy: mean = {} ,std = {}".format(np.mean(accs), np.std(accs)))
    print("5 fold Test Sens: mean = {} ,std = {}".format(np.mean(senss), np.std(senss)))
    print("5 fold Test Specs: mean = {} ,std = {}".format(np.mean(specs), np.std(specs)))



if __name__ == "__main__":

    run_kfold('../csvfiles/ukbb_5000_age.csv','AAL')
