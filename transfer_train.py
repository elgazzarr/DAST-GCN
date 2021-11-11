import torch
import numpy as np
import argparse
import time
import util
from transfer_engine import trainer
from dataset import prepare_data, prepare_fold, prepare_fold_s20
from sklearn.model_selection import KFold, StratifiedKFold
import copy
import os
import wandb
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:1',help='')
parser.add_argument('--df',type=str,default='./csvfiles/Metarest_final.csv',help='path to csv file')
parser.add_argument('--N',type=int,default=5000)
parser.add_argument('--label',type=str,default='Diagnosis')
parser.add_argument('--cv',default=True,help='5fold cross validation')
parser.add_argument('--gcn_bool',default=True,help='whether to add graph convolution layer')
parser.add_argument('--aptonly',default=True,help='whether only adaptive adj')
parser.add_argument('--dynamic',default=False,help='whether only adaptive adj')
parser.add_argument('--n_blocks',type=int,default=3,help='number of st-gcn blocks')
parser.add_argument('--nhid',type=int,default=16,help='number of filtes in hidden layers')
parser.add_argument('--kernel',type=int,default=3,help='kernel size of 1D CNNs')
parser.add_argument('--in_dim',type=int,default=3,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=112,help='number of nodes')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.5,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=200 ,help='')
parser.add_argument('--seed',type=int,default=2,help='random seed')
parser.add_argument('--transfer_model',type=str, help='path to model trained on ukbb')
parser.add_argument('--save',type=str ,help='save path')
args = parser.parse_args()
wandb.init('')


def train(train_loader,val_loader,test_loader,supports,fold):

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)
    supports = util.sym_adj(supports)
    supports = torch.tensor(supports).float().to(device)

    engine = trainer(dynamic=args.dynamic, in_dim=args.in_dim, kernel=args.kernel, num_nodes=args.num_nodes, filters=args.nhid , dropout=args.dropout, lrate=args.lr, wdecay=args.weight_decay, device=device, supports=supports, blocks=args.n_blocks,transfer=None)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    early_stop = 15
    es_counter = 0

    wandb.watch(engine.model)

    for i in range(1, args.epochs + 1):
        engine.scheduler.step()
        train_loss = 0
        t1 = time.time()

        for iter, (x, y) in enumerate(train_loader):
            trainx = x.float().to(device)
            trainx = trainx.transpose(1, 3)
            trainy = y.float().to(device)
            metrics = engine.train(trainx, trainy)
            train_loss += metrics

        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = 0

        s1 = time.time()
        for iter, (x, y) in enumerate(val_loader):
            testx = x.float().to(device)
            testx = testx.transpose(1, 3)
            testy = y.float().to(device)
            metrics = engine.eval(testx, testy)
            valid_loss += metrics
        s2 = time.time()
        val_time.append(s2 - s1)
        mtrain_loss = train_loss / len(train_loader)
        mvalid_loss = valid_loss / len(val_loader)
        his_loss.append(mvalid_loss)
        wandb.log({'train_loss_{}'.format(fold): mtrain_loss, 'valid_loss-{}'.format(fold):mvalid_loss})

        if mvalid_loss <= min(his_loss):
            print('Best val_loss = {:.4f}'.format(mvalid_loss))
            es_counter = 0
            best_model = copy.deepcopy(engine.model)
            torch.save(engine.model.state_dict(),
                       args.save + "best_fold-" + str(fold) + ".pth")
        else:
            es_counter += 1

        if es_counter > early_stop:
            print('No loss improvment.')
            break

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}'
        print(log.format(i, mtrain_loss, mvalid_loss), flush=True)


    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(args.save + "best_fold-" + str(fold) + ".pth"))

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    train_results = util.inference(engine, train_loader, args.device)
    test_results = util.inference(engine, test_loader, args.device)

    print('Fold {} Test results: '.format(fold), test_results)
    print('-' * 30)

    return test_results


def run_exp():
    results_df = pd.DataFrame(columns=['Auc','Acc','Prec','Recall'])
    k = 0

    if args.cv:
        df = pd.read_csv(args.df)
        df = df[df['Ntime']>150]
        #df = df.groupby('Sex').apply(lambda x: x.sample(n=60)).reset_index(drop=True)

        print(len(df),np.unique(df.Diagnosis.values,return_counts=True))

        kf = StratifiedKFold(n_splits=5)
        kf.get_n_splits(df, df[args.label])
        for train_index, test_index in kf.split(df, df[args.label]):

            train_loader, val_loader, test_loader,supports = prepare_fold_s20(train_index, test_index,df, args.in_dim, args.label)
            results = train(train_loader, val_loader, test_loader,supports,k)
            results_df.loc[k] = results
            k+=1


    else:
        train_loader, val_loader, test_loader = prepare_data(args.N, args.in_dim, args.label)
        results = train(train_loader, val_loader, test_loader)
        results_df.loc[k] = results

    results_df.to_csv(args.save+'results.csv')
    print('-'*50)
    print("Test Reults:")
    print('Auc = {:.4f}, {:.5f}'.format(np.mean(results_df.Auc.values), np.std(results_df.Auc.values)))
    print('Acc = {:.4f}, {:.5f}'.format(np.mean(results_df.Acc.values), np.std(results_df.Acc.values)))
    print('Prec = {:.4f}, {:.5f}'.format(np.mean(results_df.Prec.values), np.std(results_df.Prec.values)))
    print('Recall = {:.4f}, {:.5f}'.format(np.mean(results_df.Recall.values), np.std(results_df.Recall.values)))




if __name__ == "__main__":
    if not(os.path.exists(args.save)):
        os.mkdir(args.save)
    t1 = time.time()
    run_exp()
    t2 = time.time()
    #print("Total time spent: {:.4f}".format(t2-t1))
