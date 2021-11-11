import torch
import numpy as np
import argparse
import time
import util
from engine import trainer_regress
from data import prepare_data, prepare_data_handness, prepare_data_mdd, prepare_data_age, prepare_regress
import os
import shutil
import ray
from ray import tune
ray.init(num_cpus=10,num_gpus=2)
import warnings
warnings.filterwarnings('ignore')


def train(args):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)

    # load data

    device = torch.device('cuda:0')
    train_loader, val_loader, test_loader, adj_mx, seq_len, num_nodes = prepare_regress(n=args.n_samples, dim=args.in_dim, bs=args.batch_size)
    engine = trainer_regress(in_dim=args.in_dim, kernel=args.kernel, pool=args.pool, res=args.res, num_nodes=num_nodes, filters=args.n_filters , dropout=args.dropout, lrate=args.lr, wdecay=args.weight_decay, device=args.device, supports=None, blocks=args.blocks)

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    early_stop = 7
    es_counter = 0

    for i in range(1,args.epochs+1):
        if i % 10 == 0:
            lr = max(0.00001,args.lr * (0.8 ** (i // 10)))
            for g in engine.optimizer.param_groups:
                g['lr'] = lr
        train_loss = 0
        t1 = time.time()

        for iter, (x, y) in enumerate(train_loader):

            trainx = x.float().to(device)
            trainx= trainx.transpose(1, 3)
            trainy = y.float().to(device)
            metrics = engine.train(trainx, trainy)
            train_loss +=metrics

        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = 0

        s1 = time.time()
        for iter, (x, y) in enumerate(val_loader):
            testx = x.float().to(device)
            testx = testx.transpose(1, 3)
            testy = y.float().to(device)
            metrics = engine.eval(testx, testy)
            valid_loss +=metrics
        s2 = time.time()
        val_time.append(s2-s1)
        mtrain_loss = train_loss/len(train_loader)
        mvalid_loss = valid_loss/len(val_loader)
        his_loss.append(mvalid_loss)

        if mvalid_loss <= min(his_loss):
                print('Best val_loss = {:.4f}'.format(mvalid_loss))
                es_counter = 0
        else:
            es_counter += 1

        if es_counter>early_stop:
            print('No loss improvment.')
            break

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}'
        print(log.format(i, mtrain_loss, mvalid_loss),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))




    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))
    print('-'*30)

    train_results = util.inference_regress(engine,train_loader,args.device)
    test_results = util.inference_regress(engine,test_loader,args.device)

    print('Training results: ', train_results)
    print('Test results: ', test_results)

    return train_results, test_results, his_loss[bestid]


def run_search(config, checkpoint_dir=None):
    args = util.Arguments()

    shutil.rmtree(args.save)
    os.mkdir(args.save)

    args.blocks = config['n_layers']
    args.kernel = config['kernel']
    args.lr = config['lr']
    args.n_filters = config['filters']
    args.dropout = config['dropout']

    tr_results, test_results, min_val = train(args)

    tune.report(
                R2=test_results['R2'],
                Mse=test_results['MSE'],
                Mae=test_results['MAE'],
                loss=min_val)



if __name__ == "__main__":

    params = {
        "lr": tune.loguniform(1e-3, 1e-2),
        'n_layers': tune.choice([1,2, 2,3,4,5]),
        'filters': tune.choice([4, 6, 8, 16]),
        'kernel': tune.choice([3, 5, 6, 7, 9]),
        'dropout': tune.choice([0, 0.2, 0.5]),

    }

    time.sleep(30)
    analysis = tune.run(run_search, name='GWnet_ageCont',
                        config=params,
                        num_samples=200,
                        resources_per_trial={"cpu": 10, "gpu": 1})
    print(analysis.get_best_config(metric='R2'))
    print(analysis.get_best_trial(metric='R2'))



