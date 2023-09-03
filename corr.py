import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchdiffeq import odeint as odeint
import utils
import models
import scipy
from sklearn.metrics import r2_score
from models import OdeNet,DiffeqSolver, ODEFunc
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_features = 64
ode_func_net = OdeNet(n_features, 64)

rec_ode_func = ODEFunc(ode_func_net=ode_func_net)
ode_solver = DiffeqSolver(rec_ode_func, "euler", odeint_rtol=1e-3, odeint_atol=1e-4)
model = models.LOBRM('ode', 60, WS=True, HC=True, ES=True, device=device,diffeq_solver=ode_solver,
                     size_latent=64, size_latent_WS=16, time=False, n_labels=4).to(device)

five_folds_stat = True
std_label = []
loss_ls = []
fig = plt.figure()
marker = ['o','x','^']
color = ['green','red']
for i,dataset in enumerate(['INTC','MSFT','JPM']):
    for j,side in enumerate(['ask','bid']):
        record_list = torch.load('./parsed_data_/data_{}_{}_ws005_test_1day_1interval_timezscore_together_4exactlabel_sparse_100_new.pt'.format(side,dataset))
        data_obj = utils.parse_datasets(device,batch_size=512,dataset_train=None,dataset_val=None,dataset_test=record_list,train_mode=False)
        for seed in range(1):
            model.load_state_dict(torch.load('./checkpoints/{}_{}_ode_{}_True_True.ckpt'.format(dataset,side,seed))['state_dict'])
            label_predictions_ls = None
            label_real_ls = None

            for m in range(data_obj['n_test_batches']):

                batch = utils.get_next_batch(data_obj['test_dataloader'])
                extra_info = model.get_reconstruction(batch['data'], batch['time_steps'], batch['mask'], side=side, time=False)
                label_prediction = extra_info['label_predictions'].cpu().detach().numpy()
                labels = batch['labels'].cpu().detach().numpy()

                if m == 0:
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
                delta = np.mean(abs(label_real_ls - label_predictions_ls), axis=(1,2))
                plt.scatter(np.std(label_real_ls,axis=(1,2)),delta,marker=marker[i],color=color[j],label='{} {}'.format(dataset,side))
plt.xlabel('volume standard deviation',fontsize=12)
plt.ylabel('test loss',fontsize=12)
plt.legend()
plt.title('Relation between volume volatility and test loss',fontsize=16)

plt.tight_layout()
plt.show()

model = LinearRegression()
reg = model.fit(np.std(label_real_ls,axis=(1,2))[:,None],delta)
print(reg.score(np.std(label_real_ls,axis=(1,2))[:,None],delta))
