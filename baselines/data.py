import  numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.utils import shuffle

def get_corr_vector(df, corr, atlas, label, nrois=116):

    n_corr_mat = int(nrois * (nrois + 1) / 2)
    total_subjects = len(df)
    X = np.zeros((total_subjects, n_corr_mat))
    Y = np.zeros(total_subjects, dtype=int)

    if corr == 'cc':
        c = 'corrmat_file'
    elif corr == 'pc':
        c = 'partial_corrmat_file'
    else:
        c = 'mi_file'

    j = 0

    for i in range(total_subjects):

        corr_file = df[c].iloc[i].replace('ATLAS', atlas)
        corr_vals = np.load(corr_file)
        cc_triu_ids = np.triu_indices(nrois)
        cc_vector = corr_vals[cc_triu_ids]
        X[j] = cc_vector
        Y[j] = df[label].iloc[i]

        j += 1

    return X, Y


def get_corr_matrix(df, corr, atlas, nrois, label):

    df = pd.read_csv(df)
    total_subjects = len(df)
    X = np.zeros((total_subjects, nrois, nrois))
    Y = np.zeros(total_subjects, dtype=int)

    if corr == 'cc':
        c = 'corrmat_file'
    elif corr == 'pc':
        c = 'partial_corrmat_file'
    else:
        c = 'mi_file'

    j = 0

    for i in range(total_subjects):
        corr_file = df[c].iloc[i].replace('ATLAS', atlas)
        corr_vals = np.load(corr_file)
        X[j] = corr_vals
        Y[j] = df[label].iloc[i]
        j += 1

    return X, Y


class UkbbData(DataLoader):

    def __init__(self,
                 atlas_name='schaefer_400',
                 data_info_file='ukbb3000.csv',
                 z_score=True

                 ):
        super(UkbbData).__init__()

        # Check if valid atlas name
        if atlas_name not in ['AAL', 'HO_cort_maxprob_thr25-2mm', 'schaefer_100', 'schaefer_400', 'cc200', 'HO',
                              'JAMA_IC19', 'JAMA_IC52', "JAMA_IC7"]:
            raise ValueError('atlas_name not found')

        # print('Reading csv file...')
        # Read the parent CSV file
        data_info = data_info_file

        # Shuffle dataset
        data_info = shuffle(data_info, random_state=1)

        # Determine the nchannels (=nrois) from the data by using the first sample
        sample_file = data_info['tc_file'].iloc[0].replace('ATLAS', atlas_name)
        nrois = pd.read_csv(sample_file).values.shape[1] - 1
        self.nrois = nrois

        ntime = pd.read_csv(sample_file).values.shape[0]

        N_corr_mat = int(nrois * (nrois + 1) / 2)
        self.total_subjects = len(data_info)
        self.tc_data = np.zeros((self.total_subjects, nrois, ntime), dtype=float)
        labels = np.zeros(self.total_subjects, dtype=int)

        # Load data
        # print('Loading data & Creating graphs....')
        for i, sub_i in enumerate(data_info.index):
            tc_file = data_info['tc_file'].loc[sub_i].replace('ATLAS', atlas_name)
            tc_vals = pd.read_csv(tc_file).values.transpose()[1:, :ntime]

            if z_score:
                tc_vals = np.array(
                    [(tc_vals[:, i] - np.mean(tc_vals[:, i])) / np.std(tc_vals[:, i]) for i in range(tc_vals.shape[1])])
                self.tc_data[i] = tc_vals.transpose()
            else:
                self.tc_data[i] = tc_vals

            labels[i] = data_info['Age_binary'].loc[sub_i]


        # 1-hot encode it
        self.labels = np.eye(2)[labels]

    def __len__(self):
        return self.total_subjects

    def __getitem__(self, index):
        return self.tc_data[index], self.labels[index]

    def __getallitems__(self):
        return self.tc_data, self.labels


class Ukbb_corr(DataLoader):

    def __init__(self,
                 data_info_file='ukbb3000.csv',
                 atlas_name='schaefer_400'
                 ):
        super(Ukbb_corr).__init__()

        # Check if valid atlas name
        if atlas_name not in ['AAL', 'HO_cort_maxprob_thr25-2mm', 'schaefer_100', 'schaefer_400', 'cc_200', 'HO',
                              'JAMA_IC19', 'JAMA_IC52', "JAMA_IC7"]:
            raise ValueError('atlas_name not found')


        # print('Reading csv file...')
        # Read the parent CSV file
        data_info = data_info_file

        # Shuffle dataset
        data_info = shuffle(data_info, random_state=0)

        sample_file = data_info['tc_file'].iloc[0].replace('ATLAS', atlas_name)
        nrois = pd.read_csv(sample_file).values.shape[1] - 1
        N_corr_mat = int(nrois * (nrois + 1) / 2)
        self.nrois = N_corr_mat

        # Initialize an np array to store all timecourses and labels
        self.total_subjects = len(data_info)
        self.corr_data = np.zeros((self.total_subjects, N_corr_mat))
        self.graphs = []
        labels = np.zeros(self.total_subjects, dtype=int)

        # Load data
        # print('Loading data & Creating graphs....')
        for i, sub_i in enumerate(data_info.index):
            corr_file = data_info['corrmat_file'].loc[sub_i].replace('ATLAS', atlas_name)
            corr_vals = np.load(corr_file)
            cc_triu_ids = np.triu_indices(nrois)
            cc_vector = corr_vals[cc_triu_ids]
            self.corr_data[i] = cc_vector

            labels[i] = data_info['Age_binary'].loc[sub_i]

        self.labels = np.eye(2)[labels]

    def __len__(self):
        return self.total_subjects

    def __getitem__(self, index):
        return self.corr_data[index], self.labels[index]

    def __getallitems__(self):
        return self.corr_data, self.labels

class Ukbb_brainetcnn(DataLoader):

    def __init__(self,
                 data_info_file='ukbb3000.csv',
                 atlas_name='schaefer_400'
                 ):
        super(Ukbb_corr).__init__()

        # Check if valid atlas name
        if atlas_name not in ['AAL', 'HO_cort_maxprob_thr25-2mm', 'schaefer_100', 'schaefer_400', 'cc_200', 'HO',
                              'JAMA_IC19', 'JAMA_IC52', "JAMA_IC7"]:
            raise ValueError('atlas_name not found')


        # print('Reading csv file...')
        # Read the parent CSV file
        data_info = data_info_file

        # Shuffle dataset
        data_info = shuffle(data_info, random_state=0)

        sample_file = data_info['tc_file'].iloc[0].replace('ATLAS', atlas_name)
        nrois = pd.read_csv(sample_file).values.shape[1] - 1
        N_corr_mat = int(nrois * (nrois + 1) / 2)
        self.nrois = nrois

        # Initialize an np array to store all timecourses and labels
        self.total_subjects = len(data_info)
        self.corr_data = np.zeros((self.total_subjects, nrois,nrois))
        self.graphs = []
        labels = np.zeros(self.total_subjects, dtype=int)

        # Load data
        # print('Loading data & Creating graphs....')
        for i, sub_i in enumerate(data_info.index):
            corr_file = data_info['corrmat_file'].loc[sub_i].replace('ATLAS', atlas_name)
            corr_vals = np.load(corr_file)
            self.corr_data[i] = corr_vals

            labels[i] = data_info['Age_binary'].loc[sub_i]

        self.labels = np.eye(2)[labels]
        self.corr_data = np.expand_dims(self.corr_data,1)

    def __len__(self):
        return self.total_subjects

    def __getitem__(self, index):
        return self.corr_data[index], self.labels[index]

    def __getallitems__(self):
        return self.corr_data, self.labels