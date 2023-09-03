import numpy as np
import pandas as pd
import torch
from scipy.stats.mstats import winsorize
from utils import time_parser
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

for stock in ['MSFT','INTC','JPM']:
    for side in ['ask','bid']:

        record_train_list = []
        record_val_list = []
        record_test_list = []

        mean_list = []
        var_list_1 = []
        var_list_2 = []
        mean_list_tick = []
        var_list_tick = []
        count_list = []
        count_time_list = []
        statistics = True

        # for m in range(4):
        #     if m < 5:
        #         name = stock
        #         day_index = m + 1
        #     path_lob_train = './LOBSTER/%s_orderbook_part_%s.csv'%(name,day_index)
        #     path_msb_train = './LOBSTER/%s_message_part_%s.csv'%(name,day_index)
        #     lob, transaction = time_parser(path_lob_train, path_msb_train)
        #     selected = np.array(lob.iloc[:,[1,5,9,13,17]])
        #     selected = winsorize(selected,limits=[0.005,0.005])
        #     lob.iloc[:,[1,5,9,13,17]] = np.round(0.5+selected / 100.0)
        #     selected = np.array(lob.iloc[:,[3,7,11,15,19]])
        #     selected = winsorize(selected,limits=[0.005,0.005])
        #     lob.iloc[:,[3,7,11,15,19]] = np.round(0.5+selected / 100.0)
        #     transaction = transaction.astype('float64')
        #     selected = np.array(transaction.iloc[:,2])
        #     selected = winsorize(selected,limits=[0.005,0.005])
        #     transaction.iloc[:,2] = np.round(0.5+selected / 100.0)
        #     delta_time_array = np.array(lob.iloc[1:,-1]) - np.array(lob.iloc[:-1,-1])
        #
        #     if side == 'ask':
        #         mean_list.append(np.sum((delta_time_array.reshape(-1,1).repeat(6,1) * np.array(lob.iloc[:-1,[1,3,5,9,13,17]])),axis=0))
        #     else:
        #         mean_list.append(np.sum((delta_time_array.reshape(-1,1).repeat(6,1) * np.array(lob.iloc[:-1,[1,3,7,11,15,19]])),axis=0))
        #
        #     mean_list_tick.append(np.mean(np.array(transaction.iloc[:,2]))*len(lob))
        #     var_list_tick.append(np.var(np.array(transaction.iloc[:,2]))*(len(lob)))
        #     count_list.append(len(lob))
        #     count_time_list.append(sum(delta_time_array))
        #
        # mean_at_each_level = np.sum(np.array(mean_list),axis=0)/sum(count_time_list)
        # mean_at_each_level_tick = np.sum(np.array(mean_list_tick))/sum(count_list)
        # std_at_each_level_tick = np.sqrt(np.sum(np.array(var_list_tick))/(sum(count_list)))
        #
        # for m in range(4):
        #     if m < 5:
        #         name = stock
        #         day_index = m + 1
        #     path_lob_train = './LOBSTER/%s_orderbook_part_%s.csv'%(name,day_index)
        #     path_msb_train = './LOBSTER/%s_message_part_%s.csv'%(name,day_index)
        #     lob, transaction = time_parser(path_lob_train, path_msb_train)
        #     selected = np.array(lob.iloc[:,[1,5,9,13,17]])
        #     selected = winsorize(selected,limits=[0.005,0.005])
        #     lob.iloc[:,[1,5,9,13,17]] = np.round(0.5+selected / 100.0)
        #     selected = np.array(lob.iloc[:,[3,7,11,15,19]])
        #     selected = winsorize(selected,limits=[0.005,0.005])
        #     lob.iloc[:,[3,7,11,15,19]] = np.round(0.5+selected / 100.0)
        #     transaction = transaction.astype('float64')
        #     selected = np.array(transaction.iloc[:,2])
        #     selected = winsorize(selected,limits=[0.005,0.005])
        #     transaction.iloc[:,2] = np.round(0.5+selected / 100.0)
        #     delta_time_array = np.array(lob.iloc[1:,-1]) - np.array(lob.iloc[:-1,-1])
        #     var_list_1.append(np.sum(np.square(np.array(lob.iloc[:-1,[1,3]]) - np.mean(mean_at_each_level[:2])) * delta_time_array.reshape(-1,1).repeat(2,1),axis=0))
        #     if side == 'ask':
        #         var_list_2.append(np.sum(np.square(np.array(lob.iloc[:-1,[5,9,13,17]]) - np.mean(mean_at_each_level[2:])) * delta_time_array.reshape(-1,1).repeat(4,1),axis=0))
        #     else:
        #         var_list_2.append(np.sum(np.square(np.array(lob.iloc[:-1,[7,11,15,19]]) - np.mean(mean_at_each_level[2:])) * delta_time_array.reshape(-1,1).repeat(4,1),axis=0))
        # std_at_each_level_1 = np.sqrt(np.sum(np.array(var_list_1))/np.sum(count_time_list)/2)
        # std_at_each_level_2 = np.sqrt(np.sum(np.array(var_list_2))/np.sum(count_time_list)/4)
        #
        # mean_at_each_level_1 = np.mean(mean_at_each_level[:2])
        # mean_at_each_level_2 = np.mean(mean_at_each_level[2:])
        #
        # np.save('./statistics/{}_{}_mean_at_each_level.npy'.format(stock,side),mean_at_each_level)
        # np.save('./statistics/{}_{}_mean_at_each_level_tick.npy'.format(stock,side),mean_at_each_level_tick)
        # np.save('./statistics/{}_{}_std_at_each_level_tick.npy'.format(stock,side),std_at_each_level_tick)
        # np.save('./statistics/{}_{}_std_at_each_level_1.npy'.format(stock,side),std_at_each_level_1)
        # np.save('./statistics/{}_{}_std_at_each_level_2.npy'.format(stock,side),std_at_each_level_2)
        # np.save('./statistics/{}_{}_mean_at_each_level_1.npy'.format(stock,side),mean_at_each_level_1)
        # np.save('./statistics/{}_{}_mean_at_each_level_2.npy'.format(stock,side),mean_at_each_level_2)

        mean_at_each_level = np.load('./statistics/{}_{}_mean_at_each_level.npy'.format(stock,side))
        mean_at_each_level_tick = np.load('./statistics/{}_{}_mean_at_each_level_tick.npy'.format(stock,side))
        std_at_each_level_tick = np.load('./statistics/{}_{}_std_at_each_level_tick.npy'.format(stock,side))
        std_at_each_level_1 = np.load('./statistics/{}_{}_std_at_each_level_1.npy'.format(stock,side))
        std_at_each_level_2 = np.load('./statistics/{}_{}_std_at_each_level_2.npy'.format(stock,side))
        mean_at_each_level_1 = np.load('./statistics/{}_{}_mean_at_each_level_1.npy'.format(stock,side))
        mean_at_each_level_2 = np.load('./statistics/{}_{}_mean_at_each_level_2.npy'.format(stock,side))

        lob_ls = []
        for m in range(4):
            name = stock
            day_index = m + 1
            path_lob_train = './LOBSTER/%s_orderbook_part_%s.csv'%(name,day_index)
            path_msb_train = './LOBSTER/%s_message_part_%s.csv'%(name,day_index)
            lob, transaction = time_parser(path_lob_train, path_msb_train)
            lob_ls.append(np.array(lob))
        lob_ls = np.vstack(lob_ls)
        p_scaler = StandardScaler()
        p_scaler.fit(lob_ls[:,list(range(0,20,2))])

        for m in range(5):
            if m < 5:
                name = stock
                day_index = m + 1
            path_lob_train = './LOBSTER/%s_orderbook_part_%s.csv'%(name,day_index)
            path_msb_train = './LOBSTER/%s_message_part_%s.csv'%(name,day_index)
            lob, transaction = time_parser(path_lob_train, path_msb_train)
            selected = np.array(lob.iloc[:,[1,5,9,13,17]])
            selected = winsorize(selected,limits=[0.005,0.005])
            lob.iloc[:,[1,5,9,13,17]] = np.round(0.5+selected / 100.0)
            selected = np.array(lob.iloc[:,[3,7,11,15,19]])
            selected = winsorize(selected,limits=[0.005,0.005])
            lob.iloc[:,[3,7,11,15,19]] = np.round(0.5+selected / 100.0)
            transaction = transaction.astype('float64')
            selected = np.array(transaction.iloc[:,2])
            selected = winsorize(selected,limits=[0.005,0.005])
            transaction.iloc[:,2] = np.round(0.5+selected / 100.0)
            transaction.iloc[:,2] = (transaction.iloc[:,2]-mean_at_each_level_tick) / std_at_each_level_tick
            lob.iloc[:, 1] = (lob.iloc[:, 1] - mean_at_each_level_1) / std_at_each_level_1
            lob.iloc[:, 3] = (lob.iloc[:, 3] - mean_at_each_level_1) / std_at_each_level_1
            lob.iloc[:,list(range(5,21,2))] = (lob.iloc[:,list(range(5,21,2))] - mean_at_each_level_2) / std_at_each_level_2
            lob.iloc[:,list(range(0,20,2))] = p_scaler.transform(lob.iloc[:,list(range(0,20,2))])

            num_transaction = len(transaction)
            max_len = 100
            position = max_len
            index = 0
            device = torch.device('cuda:0')
            while position < num_transaction:
                target_1 = lob[[0,1,2,3,'time']][position-max_len:position]
                target_2 = transaction.iloc[position-max_len:position,:]
                time = torch.FloatTensor((target_1['time']).values-target_1.iloc[0,4])[-max_len:]
                data = np.zeros(shape = (max_len,7))

                '''for explicit encoding'''
                for i in range(max_len):
                    for j in range(7):
                        if j<4:
                            data[i,j] = target_1.iloc[i,j]
                        else:
                            data[i,j] = target_2.iloc[i,j-3]
                data_encoded = torch.FloatTensor(data).to(device)
                mask = np.ones(shape=(max_len, 7))
                mask = torch.FloatTensor(mask).to(device)

                '''for sparse encoding'''
                # for i in range(max_len):
                #     for j in range(7):
                #         if j < 4:
                #             data[i,j] = target_1.iloc[i,j]
                #         else:
                #             data[i,j] = target_2.iloc[i,j-3]
                # for i in range(max_len):
                #     if data[i,6] == -1:
                #         data[i,4] = (data[i,4] - data[-1,0])/100
                #     elif data[i,6] == 1:
                #         data[i,4] = (data[i,4] - data[-1,2])/100
                # data[:,[0,2]] = (data[:,[0,2]] - data[-1,[0,2]])/100
                # ask_encoder = np.zeros(shape=(max_len,15))
                # bid_encoder = np.zeros(shape=(max_len,15))
                # tik_ask_encoder = np.zeros(shape=(max_len,15))
                # tik_bid_encoder = np.zeros(shape=(max_len,15))
                #
                # for i in range(max_len):
                #     if 7+int(data[-(i+1),0])>=0 and 7+int(data[-(i+1),0])<=14:
                #         ask_encoder[-(i+1),7+int(data[-(i+1),0])] = data[-(i+1),1]
                #         if data[-(i+1),6] == -1: # market buy happening at ask side
                #             try:
                #                 tik_ask_encoder[-(i+1),7+int(data[-(i+1),4])]=-data[-(i+1),5]
                #             except Exception as ex:
                #                 print(ex)
                #
                #     if 7+int(data[-(i+1),2])>=0 and 7+int(data[-(i+1),2])<=14:
                #         bid_encoder[-(i+1),7+int(data[-(i+1),2])] = data[-(i+1),3]
                #         if data[-(i+1),6] == 1:
                #             try:
                #                 tik_bid_encoder[-(i+1),7+int(data[-(i+1),4])]=-data[-(i+1),5]
                #             except Exception as ex:
                #                 print(ex)
                #
                # data_encoded = np.hstack((ask_encoder,bid_encoder,tik_ask_encoder,tik_bid_encoder))
                # mask = np.ones(shape = (max_len,60)) - (np.zeros(shape = (max_len,60)) == data_encoded)
                # mask = torch.FloatTensor(mask).to(device)
                # data_encoded = torch.FloatTensor(data_encoded).to(device)

                label = np.zeros(shape = 4)
                q_vec_ask = lob.iloc[position-1,[1,5,9,13,17]]
                q_vec_bid = lob.iloc[position-1,[3,7,11,15,19]]
                for i in range(4):
                    if side == 'ask':
                        label[i] = q_vec_ask[4*i+5]
                    else:
                        label[i] = q_vec_bid[4*i+7]
                label = torch.FloatTensor(label).to(device)

                if m == 3:
                    record_val_list.append((index,time,data_encoded,mask,label))
                elif m == 4:
                    record_test_list.append((index,time,data_encoded,mask,label))
                else:
                    record_train_list.append((index, time, data_encoded, mask, label))
                position = position + 1
                index = index + 1

        torch.save(record_train_list,'./parsed_data_/data_{}_{}_ws005_train_3days_1interval_timezscore_together_4exactlabel_explicit_100_3_new.pt'.format(side,stock))
        torch.save(record_val_list,'./parsed_data_/data_{}_{}_ws005_val_1day_1interval_timezscore_together_4exactlabel_explicit_100_3_new.pt'.format(side,stock))
        torch.save(record_test_list,'./parsed_data_/data_{}_{}_ws005_test_1day_1interval_timezscore_together_4exactlabel_explicit_100_new.pt'.format(side,stock))
