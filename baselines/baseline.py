import  numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import logging
import numpy
from sklearn.model_selection import KFold, StratifiedKFold
import sklearn.metrics as results

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, ParameterGrid
from sklearn.svm import SVC
from scipy.stats import reciprocal, uniform
import matplotlib.pyplot as plt

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



class KernelSVC_Model:

    def __init__(self, kernel='linear'):

        self.kernel = kernel

    def get_model(self):

        return SVC(kernel=self.kernel, max_iter=1000)

    def run(self, x_train, y_train, x_test, y_test):
        model = self.get_model()
        if self.kernel == 'rbf':
            params = {'gamma': [1e-2, 1e-3, 1e-4],
                      'C': [1, 10, 100, 1000]}
        else:
            params = {'C': [1, 10, 100, 1000]}
        rnd_search_cv = RandomizedSearchCV(model, params, n_iter=12, cv=3, random_state=0)
        n_samples = x_train.shape[0]


        cv_samples = int(0.5*n_samples)

        rnd_search_cv.fit(x_train[:cv_samples],y_train[:cv_samples])
        rnd_search_cv.best_estimator_.fit(x_train,y_train)
        y_pred = rnd_search_cv.best_estimator_.predict(x_test)

        return {"Acc": round(results.balanced_accuracy_score(y_test, y_pred), 3),
                "Prec": round(results.precision_score(y_test, y_pred), 3),
                "Recall": round(results.recall_score(y_test, y_pred), 3)}


def run_main(kernel='rbf',conn='cc',atlas='AAL',label='Age_binary'):
    results_df = pd.DataFrame(columns=['Acc','Prec','Recall'])
    k = 0
    df = pd.read_csv('../csvfiles/ukbb_10k.csv')
    df['Sex'].replace({'Male':0, 'Female':1},inplace=True)

    print(np.unique(df['Sex'],return_counts=True))
    print(np.mean(df["Age"]),np.std(df['Age']))
    kf = StratifiedKFold(n_splits=5)
    kf.get_n_splits(df,df[label])
    for train_index, test_index in kf.split(df,df[label]):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        x_train, y_train = get_corr_vector(df_train, conn, atlas, label)
        x_test, y_test = get_corr_vector(df_test, conn, atlas, label)

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        model = KernelSVC_Model(kernel=kernel)

        r = model.run(x_train, y_train, x_test, y_test)
        print('fold {}:'.format(k), r)
        results_df.loc[k] = r
        k += 1
    #results_df.to_csv('svm_linear_age.csv')
    print('-'*50)
    print("Test Results:")
    print('Acc = {:.4f}, {:.5f}'.format(np.mean(results_df.Acc.values), np.std(results_df.Acc.values)))
    print('Prec = {:.4f}, {:.5f}'.format(np.mean(results_df.Prec.values), np.std(results_df.Prec.values)))
    print('Recall = {:.4f}, {:.5f}'.format(np.mean(results_df.Recall.values), np.std(results_df.Recall.values)))




if __name__ == "__main__":

    run_main()
