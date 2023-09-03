import numpy as np
import pandas as pd
import sys
import tarfile
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import logging
import utils
import models
import argparse
import os
from models import OdeNet,DiffeqSolver, ODEFunc

def define_args():
    parser = argparse.ArgumentParser('LOB')
    parser.add_argument('--dataset',  type=str, default="INTC", help="dataset for the source model")
    parser.add_argument('--side', type=str, default="bid", help="ask side model or bid side model")
    parser.add_argument('--bs', type=int, default=512, help="batch size")
    parser.add_argument('--niter', type=int, default=50, help="number of iterations")
    parser.add_argument('--lr',  type=float, default=1e-3, help="Starting learning rate")
    parser.add_argument('--ls',  type=int, default=64, help="latent size in HC and ES")
    parser.add_argument('--ls_weight',  type=int, default=16, help="latent size in WS")
    parser.add_argument('--validate',  type=int, default=3, help="position of validate")

    parser.add_argument('--n_labels',  type=int, default=4, help="number of labels")
    parser.add_argument('--n_units',  type=int, default=64, help="number of units in all MLPs")
    parser.add_argument('--seed',  type=int, default=0, help="random seed")

    parser.add_argument('--main_module', type=str, default='attention', help="use what module")
    parser.add_argument('--time', action='store_true', default=False, help="whether to add time to feature vector or not")

    parser.add_argument('--WS', type=bool, default=False, help="whether to use weighting scheme or not")
    parser.add_argument('--HC', type=bool, default=False, help="whether to use history compiler or not")
    parser.add_argument('--ES', type=bool, default=True, help="whether to use market events simulator or not")
    return parser.parse_args()

def main(dataset='MSFT',side='bid',main_module='ode',HC=False,seed=0,gpu=0):

    args = define_args()
    args.dataset = dataset
    args.side = side
    args.main_module = main_module
    args.seed = seed
    args.HC = HC
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = "./checkpoints/{}_{}_{}_{}_{}_{}_ex.ckpt".format(args.dataset,args.side,args.main_module,args.seed,args.ES,args.HC)
    log_path = "./logs/{}_{}_{}_{}_{}_{}_ex.log".format(args.dataset,args.side,args.main_module,args.seed,args.ES,args.HC)
    logger = open(log_path, "w")

    record_list_train = torch.load('./parsed_data_/data_%s_%s_ws005_train_3days_1interval_timezscore_together_4exactlabel_explicit_100_%d_new.pt' %(args.side,args.dataset,args.validate))
    record_list_val = torch.load('./parsed_data_/data_%s_%s_ws005_val_1day_1interval_timezscore_together_4exactlabel_explicit_100_%d_new.pt' % (args.side, args.dataset,args.validate))
    record_list_test = torch.load('./parsed_data_/data_%s_%s_ws005_test_1day_1interval_timezscore_together_4exactlabel_explicit_100_new.pt' %(args.side,args.dataset))
    data_obj = utils.parse_datasets(device,batch_size=args.bs,dataset_train = record_list_train, dataset_val = record_list_val, dataset_test = record_list_test, train_mode = True)
    input_dim = data_obj["input_dim"]
    num_batches = data_obj["n_train_batches"]

    if args.main_module == 'ode':

        n_features = 64
        ode_func_net = OdeNet(n_features, 64)

        rec_ode_func = ODEFunc(ode_func_net=ode_func_net)
        ode_solver = DiffeqSolver(rec_ode_func, "euler", odeint_rtol=1e-3, odeint_atol=1e-4)
        model = models.LOBRM(args.main_module, input_dim, WS = args.WS, HC = args.HC, ES = args.ES, device=device, diffeq_solver=ode_solver,
                               size_latent= args.ls, size_latent_WS=args.ls_weight, time=args.time, n_labels=args.n_labels).to(device)

    else:
        model = models.LOBRM(args.main_module, input_dim, WS = args.WS, HC = args.HC, ES = args.ES, device=device, diffeq_solver=None,
                               size_latent= args.ls, size_latent_WS=args.ls_weight, time=args.time, n_labels=args.n_labels).to(device)

    train_loss = 0
    train_samples = 0
    val_loss_list = []
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    for itr in range(1, num_batches * (args.niter+1)):
        optimizer.zero_grad()
        batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
        train_res = model.compute_all_losses(batch_dict, args.side, args.time)
        train_res["l1_loss"].backward()
        optimizer.step()
        train_loss = train_loss + train_res['l1_loss']*len(batch_dict['data'])
        train_samples = train_samples + len(batch_dict['data'])
        n_iters_to_val = 1
        if itr % (n_iters_to_val * num_batches) == 0:
            with torch.no_grad():
                val_res = utils.compute_loss_all_batches(model, data_obj["val_dataloader"],
                                                          n_batches=data_obj["n_val_batches"],side = args.side,time= args.time)
                logger.write("Epoch {:04d}\n".format(itr//num_batches))
                logger.write("Train l1 loss: {:.6f}\n".format((train_loss / train_samples).detach()))
                logger.write('Validation l1 Loss {:.6f}\n'.format(val_res["l1_loss"].detach()))
                print("Train l1 loss: {:.6f}".format((train_loss / train_samples).detach()))
                print('Validation l1 Loss {:.6f}'.format(val_res["l1_loss"].detach()))
                train_loss = 0
                train_samples = 0
                val_loss_list.append(val_res['l1_loss'])

                if val_loss_list[-1] == min(val_loss_list):
                    torch.save({'state_dict': model.state_dict(),}, ckpt_path)

    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    with torch.no_grad():
        val_res = utils.compute_loss_all_batches(model, data_obj["val_dataloader"],
                                                  n_batches=data_obj["n_val_batches"],side = args.side,time= args.time)
        message = 'Final Val l1 Loss {:.6f}\n'.format(val_res["l1_loss"].detach())
        logger.write(message)
        print('Final Val l1 Loss {:.6f}'.format(val_res["l1_loss"].detach()))
        test_res = utils.compute_loss_all_batches(model, data_obj["test_dataloader"],
                                                  n_batches=data_obj["n_test_batches"],side = args.side,time= args.time)
        message = 'Final Test l1 Loss {:.6f}\n'.format(test_res["l1_loss"].detach())
        logger.write(message)
        print('Final Test l1 Loss {:.6f}'.format(test_res["l1_loss"].detach()))
        logger.close()


if __name__ == '__main__':
    main()
