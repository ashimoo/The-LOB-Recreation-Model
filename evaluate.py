import numpy as np
import pandas as pd
import os
import sys
import tarfile
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchdiffeq import odeint as odeint
import torch.optim as optim
import logging
import utils
import models
import argparse
import scipy
from numpy import mean
from sklearn.metrics import r2_score
from models import OdeNet,DiffeqSolver, ODEFunc
import torchode as to

def main():
    module = 'ode'
    dataset = 'JPM'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = 64
    ode_func_net = OdeNet(n_features, 64)

    rec_ode_func = ODEFunc(ode_func_net=ode_func_net)
    ode_solver = DiffeqSolver(rec_ode_func, "euler", odeint_rtol=1e-3, odeint_atol=1e-4)
    model = models.LOBRM(module, 60, WS=True, HC=True, ES=True, device=device,diffeq_solver=ode_solver if module=='ode' else None,
                         size_latent=64, size_latent_WS=16, time=False, n_labels=4).to(device)

    five_folds_stat = False

    for side in ['ask','bid']:
        record_list = torch.load('./parsed_data_/data_{}_{}_ws005_test_1day_1interval_timezscore_together_4exactlabel_sparse_100_new.pt'.format(side,dataset))
        data_obj = utils.parse_datasets(device,batch_size=512,dataset_train=None,dataset_val=None,dataset_test=record_list,train_mode=False)
        r_list = []
        for seed in range(1):
            model.load_state_dict(torch.load('./checkpoints/{}_{}_{}_{}_True_True.ckpt'.format(dataset,side,module,seed))['state_dict'])
            label_predictions_ls = None
            label_real_ls = None

            for i in range(data_obj['n_test_batches']):

                batch = utils.get_next_batch(data_obj['test_dataloader'])
                extra_info = model.get_reconstruction(batch['data'], batch['time_steps'], batch['mask'], side=side, time=False)
                label_prediction = extra_info['label_predictions'].cpu().detach().numpy()
                labels = batch['labels'].cpu().detach().numpy()

                if i == 0:
                    label_predictions_ls = label_prediction
                    label_real_ls = labels
                else:
                    label_predictions_ls = np.vstack((label_predictions_ls,label_prediction))
                    label_real_ls = np.vstack((label_real_ls,labels))

            '''divide into 5 folds'''
            if five_folds_stat:
                folds = len(label_predictions_ls)//5
                residue = len(label_predictions_ls)%5
                if residue:
                    label_predictions_ls = label_predictions_ls[:-residue].reshape(5,folds,4)
                    label_real_ls = label_real_ls[:-residue].reshape(5,folds,4)
                else:
                    label_predictions_ls = label_predictions_ls.reshape(5,folds,4)
                    label_real_ls = label_real_ls.reshape(5,folds,4)
                label_predictions_ls = np.mean(label_predictions_ls,axis=1)
                label_real_ls = np.mean(label_real_ls,axis=1)
                delta = np.mean(abs(label_real_ls - label_predictions_ls), axis=1)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(label_predictions_ls.reshape(-1), label_real_ls.reshape(-1))
            print('R_squared value for {} side prediction is {}'.format(side,r_value**2))
            r_list.append(r_value**2)

            '''unnormalize the prediction'''
            mean = np.load('./statistics/{}_{}_mean_at_each_level_2.npy'.format(dataset,side))
            std = np.load('./statistics/{}_{}_std_at_each_level_2.npy'.format(dataset,side))
            label_predictions_ls = (label_predictions_ls * std) + mean

            if side == 'ask':
                label_predictions_ask = label_predictions_ls
            else:
                label_predictions_bid = label_predictions_ls

    path_lob_train = './LOBSTER/{}_orderbook_part_5.csv'.format(dataset)
    path_msb_train = './LOBSTER/{}_message_part_5.csv'.format(dataset)
    lob, transaction = utils.time_parser(path_lob_train, path_msb_train)

    recreated_lob = pd.DataFrame(np.zeros(shape=(len(label_predictions_ask), 21)))
    recreated_length = len(label_predictions_ask)
    for i in range(recreated_length):
        recreated_lob.iloc[-(i + 1), 20] = lob.iloc[-(i + 1), 20]
    print('Done')
    recreated_lob.iloc[:, [0, 1, 2, 3]] = lob.iloc[-recreated_length:, [0, 1, 2, 3]].values
    recreated_lob.iloc[:, [5, 9, 13, 17]] = (100 * label_predictions_ask).round()
    recreated_lob.iloc[:, 4] = recreated_lob.iloc[:, 0] + 100
    recreated_lob.iloc[:, 8] = recreated_lob.iloc[:, 0] + 200
    recreated_lob.iloc[:, 12] = recreated_lob.iloc[:, 0] + 300
    recreated_lob.iloc[:, 16] = recreated_lob.iloc[:, 0] + 400
    recreated_lob.iloc[:, [7, 11, 15, 19]] = (100 * label_predictions_bid).round()
    recreated_lob.iloc[:, 6] = recreated_lob.iloc[:, 2] - 100
    recreated_lob.iloc[:, 10] = recreated_lob.iloc[:, 2] - 200
    recreated_lob.iloc[:, 14] = recreated_lob.iloc[:, 2] - 300
    recreated_lob.iloc[:, 18] = recreated_lob.iloc[:, 2] - 400
    recreated_lob.to_csv('./fake_lob/fake_lob_{}.csv'.format(dataset), header=True)
    lob.iloc[-len(recreated_lob):,:20].to_csv('./fake_lob/real_lob_{}.csv'.format(dataset),header=True)
    print('done')
if __name__ == '__main__':
    main()
