import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split

import util



class UkbbData(DataLoader):

    def __init__(self,
                 atlas_name='schaefer_400',
                 data_info_file='ukbb3000.csv',
                 in_dim =2,
                 scaler=None,
                 train=True,
                 y='Sex',
                 n_classes=2,

                 ):
        super(UkbbData).__init__()

        # Check if valid atlas name
        if atlas_name not in ['AAL', 'HO_cort_maxprob_thr25-2mm', 'schaefer_100', 'schaefer_400', 'cc200', 'HO',
                              'JAMA_IC19', 'JAMA_IC52', "JAMA_IC7",'HO']:
            raise ValueError('atlas_name not found')

        # print('Reading csv file...')
        # Read the parent CSV file
        data_info = data_info_file

        # Shuffle dataset
        data_info = shuffle(data_info, random_state=1)

        # Determine the n-channels (=nrois) from the data by using the first sample
        sample_file = data_info['tc_file'].iloc[0].replace('ATLAS', atlas_name)
        nrois = pd.read_csv(sample_file).values.shape[1] - 1
        self.nrois = nrois
        self.ntime = 490
        self.total_subjects = len(data_info)
        self.tc_data = np.zeros((self.total_subjects, nrois, self.ntime,in_dim),dtype=float)
        self.corr_data = np.zeros((self.total_subjects, nrois, nrois))

        labels = np.zeros(self.total_subjects, dtype=int)

        # Load data
        # print('Loading data & Creating graphs....')
        for i, sub_i in enumerate(data_info.index):

            tc_file = data_info['tc_file'].loc[sub_i].replace('ATLAS', atlas_name)
            tc_vals = pd.read_csv(tc_file).values.transpose()[1:, :self.ntime]

            corr_file = data_info['corrmat_file'].loc[sub_i].replace('ATLAS', atlas_name)
            corr_vals = np.load(corr_file)
            self.corr_data[i] = corr_vals

            tc_vals = tc_vals
            tc_diff = np.array([np.pad(np.diff(tc_vals[i]),(0,1),'edge') for i in range(tc_vals.shape[0])])
            tc_diff2 = np.array([np.pad(np.diff(tc_diff[i]),(0,1),'edge') for i in range(tc_diff.shape[0])])

            self.tc_data[i,:,:,0] = tc_vals
            if in_dim>1:
                self.tc_data[i,:,:,1] = tc_diff
            if in_dim>2:
                self.tc_data[i,:,:,2] = tc_diff2


            if y=='Sex':
                if data_info[y].loc[sub_i] == 'Male':
                    labels[i] = 0
                else:
                    labels[i] = 1
            else:
                labels[i] = int(data_info[y].loc[sub_i])



        data = np.reshape(self.tc_data,(self.total_subjects,-1))

        if train:
            self.scaler = StandardScaler()
            self.scaler.fit(data)
        else:
            self.scaler = scaler

        data = self.scaler.transform(data)

        self.tc_data = np.reshape(data,(self.total_subjects,nrois,self.ntime,in_dim))



        cc_mean = np.mean(self.corr_data,axis=0)
        a = np.abs(cc_mean)
        #a, thr_calc = util.sparsing(cc_mean, 50)

        self.tc_data = np.transpose(self.tc_data, (0, 2, 1,3))
        self.labels = np.eye(n_classes)[labels]
        self.adj = a

    def __len__(self):
        return self.total_subjects

    def __getitem__(self, index):
        return self.tc_data[index], self.labels[index]

    def __getallitems__(self):
        return self.tc_data, self.labels

class mdd_S20(DataLoader):

    def __init__(self,
                 atlas_name='schaefer_400',
                 data_info_file='ukbb3000.csv',
                 in_dim =2,
                 scaler=None,
                 train=True,
                 y='Sex',
                 n_classes=2

                 ):
        super(mdd_S20).__init__()


        # print('Reading csv file...')
        # Read the parent CSV file
        data_info = data_info_file

        # Shuffle dataset
        data_info = shuffle(data_info, random_state=1)

        # Determine the n-channels (=nrois) from the data by using the first sample
        sample_file = data_info['tc_atlas'].iloc[0].replace('ATLAS', atlas_name)
        ntime, nrois = np.load(sample_file).shape
        ntime = 150#min(data_info["Ntime"])
        self.nrois = nrois
        self.ntime = ntime
        self.total_subjects = len(data_info)
        self.tc_data = np.zeros((self.total_subjects, nrois, self.ntime,in_dim),dtype=float)
        self.corr_data = np.zeros((self.total_subjects, nrois, nrois))

        labels = np.zeros(self.total_subjects, dtype=int)

        # Load data
        # print('Loading data & Creating graphs....')
        for i, sub_i in enumerate(data_info.index):

            tc_file = data_info['tc_atlas'].iloc[i].replace('ATLAS', atlas_name)
            tc_vals = np.load(tc_file).transpose()[:,:ntime]

            tc_diff = np.array([np.pad(np.diff(tc_vals[i]),(0,1),'edge') for i in range(tc_vals.shape[0])])
            tc_diff2 = np.array([np.pad(np.diff(tc_diff[i]),(0,1),'edge') for i in range(tc_diff.shape[0])])

            self.tc_data[i,:,:,0] = tc_vals
            if in_dim>1:
                self.tc_data[i,:,:,1] = tc_diff
            if in_dim>2:
                self.tc_data[i,:,:,2] = tc_diff2

            labels[i] = data_info[y].iloc[i]-1

        data = np.reshape(self.tc_data,(self.total_subjects,-1))

        if train:
            self.scaler = StandardScaler()
            self.scaler.fit(data)
        else:
            self.scaler = scaler

        data = self.scaler.transform(data)
        self.adj = np.zeros((116,116))

        self.tc_data = np.reshape(data,(self.total_subjects,nrois,self.ntime,in_dim))

        self.tc_data = np.transpose(self.tc_data, (0, 2, 1,3))
        self.labels = np.eye(2)[labels]

    def __len__(self):
        return self.total_subjects

    def __getitem__(self, index):
        return self.tc_data[index], self.labels[index]

    def __getallitems__(self):
        return self.tc_data, self.labels




def prepare_data(n, dim, lbl, atlas='AAL', bs=32):

    print('Preparing Data ...')

    if n == 'all':
        df = pd.read_csv('/data/agelgazzar/projects/DAST-GCN/csvfiles/ukbb_10k.csv')
        df_train_val, df_test = train_test_split(df,test_size=0.2,stratify=df[lbl], random_state=1)
        df_train, df_val = train_test_split(df_train_val,test_size=0.1, stratify=df_train_val[lbl], random_state=1)
    else:
        df_train_val = pd.read_csv('/data/agelgazzar/projects/DAST-GCN/csvfiles/ukbb_{}.csv'.format(n))
        df_train, df_val = train_test_split(df_train_val,test_size=0.1, stratify=df_train_val[lbl], random_state=1)
        df_test = pd.read_csv('/data/agelgazzar/projects/DAST-GCN/csvfiles/ukbb_test.csv')

    dataset = UkbbData(data_info_file=df_train, atlas_name=atlas, in_dim=dim, train=True,y=lbl)
    train_scaler = dataset.scaler
    dataset_val = UkbbData(data_info_file=df_val, atlas_name=atlas,in_dim=dim, scaler=train_scaler, train=False, y=lbl)
    dataset_test = UkbbData(data_info_file=df_test, atlas_name=atlas,in_dim=dim, scaler=train_scaler, train=False, y=lbl)

    adj = dataset.adj
    seq_len = dataset.ntime
    num_nodes = adj.shape[1]

    train_loader = DataLoader(dataset, batch_size=bs)
    val_loader = DataLoader(dataset_val, batch_size=bs)
    test_loader = DataLoader(dataset_test, batch_size=bs)

    print('Done.')
    return train_loader, val_loader, test_loader







def prepare_fold(train_index, test_index, df, dim, lbl, atlas='AAL', bs=32):

    print('Preparing Data ...')

    df_train_val = df.iloc[train_index]
    df_test = df.iloc[test_index]
    df_train, df_val = train_test_split(df_train_val, test_size=0.1, stratify=df_train_val[lbl], random_state=1)

    dataset = UkbbData(data_info_file=df_train, atlas_name=atlas, in_dim=dim, train=True,y=lbl)
    train_scaler = dataset.scaler
    dataset_val = UkbbData(data_info_file=df_val, atlas_name=atlas,in_dim=dim, scaler=train_scaler, train=False, y=lbl)
    dataset_test = UkbbData(data_info_file=df_test, atlas_name=atlas,in_dim=dim, scaler=train_scaler, train=False, y=lbl)

    adj = dataset.adj
    seq_len = dataset.ntime
    num_nodes = adj.shape[1]

    train_loader = DataLoader(dataset, batch_size=bs)
    val_loader = DataLoader(dataset_val, batch_size=bs)
    test_loader = DataLoader(dataset_test, batch_size=bs)

    print('Done.')
    return train_loader, val_loader, test_loader, adj

def prepare_fold_s20(train_index, test_index, df, dim, lbl, atlas='HO_112', bs=16):

    print('Preparing Data ...')

    df_train_val = df.iloc[train_index]
    df_test = df.iloc[test_index]
    df_train, df_val = train_test_split(df_train_val, test_size=0.1, stratify=df_train_val[lbl], random_state=1)

    dataset = mdd_S20(data_info_file=df_train, atlas_name=atlas, in_dim=dim, train=True,y=lbl)
    train_scaler = dataset.scaler
    dataset_val = mdd_S20(data_info_file=df_val, atlas_name=atlas,in_dim=dim, scaler=train_scaler, train=False, y=lbl)
    dataset_test = mdd_S20(data_info_file=df_test, atlas_name=atlas,in_dim=dim, scaler=train_scaler, train=False, y=lbl)

    adj = dataset.adj
    seq_len = dataset.ntime
    num_nodes = adj.shape[1]

    train_loader = DataLoader(dataset, batch_size=bs)
    val_loader = DataLoader(dataset_val, batch_size=bs)
    test_loader = DataLoader(dataset_test, batch_size=bs)

    print('Done.')
    return train_loader, val_loader, test_loader, adj