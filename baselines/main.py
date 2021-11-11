from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from models.linear_models import  LogisticRegression_Model, KernelSVM_Model
from data import get_corr_matrix, get_corr_vector, get_graph
import argparse
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def main(args):

    df_train = args.csv_path + 'ukbb_{}'.format(str(args.n_samples)) + '.csv'
    df_test = args.csv_path + args.df_test
    x_train, y_train = get_corr_vector(df_train, args.conn, args.atlas, args.label)
    x_test, y_test = get_corr_vector(df_test, args.conn, args.atlas, args.label)

    scaler = StandardScaler()

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    if args.model == 'logisitic-reg':
        model = LogisticRegression_Model()

    elif args.model == 'svm_rbf':
        model = KernelSVM_Model(kernel='rbf')

    elif args.model == 'svm_linear':
        model = KernelSVM_Model(kernel='linear')

    else:
        raise ValueError("Model not yet defined")

    acc, f1 = model.run(x_train, y_train, x_test, y_test)

    print('For {} model at {} samples, Test_Acc = {} ,Test_F1 = {}'.format(args.model, args.n_samples, acc, f1))


def run_onehsot(args):

    df = pd.read_csv('../csvfiles/ukbb_10k.csv')
    df[args.label].replace({'Male':0, 'Female':1},inplace=True)
    df_train, df_test = train_test_split(df,test_size=0.1, stratify=df[args.label], random_state=1)
    x_train, y_train = get_corr_vector(df_train, args.conn, args.atlas, args.label)
    x_test, y_test = get_corr_vector(df_test, args.conn, args.atlas, args.label)

    scaler = StandardScaler()

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)


    if args.model == 'logisitic-reg':
        model = LogisticRegression_Model()

    elif args.model == 'svm_rbf':
        model = KernelSVM_Model(kernel='rbf')

    elif args.model == 'svm_linear':
        model = KernelSVM_Model(kernel='linear')

    else:
        raise ValueError("Model not yet defined")

    acc, f1 = model.run(x_train, y_train, x_test, y_test)

    print('For {} model at {} samples, Test_Acc = {} ,Test_F1 = {}'.format(args.model, len(df_train), acc, f1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UKbb fmri sex Classification')
    parser.add_argument('--model', '-m', default='svm_rbf')
    parser.add_argument('--n_samples','-n', default='500')
    parser.add_argument('--csv_path', default='./csvfiles/')
    parser.add_argument('--df_test', default='ukbb_test.csv')
    parser.add_argument('--atlas', default='AAL')
    parser.add_argument('--conn', default= 'cc')
    parser.add_argument('--label', default= 'Sex')



    args = parser.parse_args()

    run_onehsot(args)
