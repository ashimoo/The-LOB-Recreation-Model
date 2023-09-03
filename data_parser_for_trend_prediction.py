
import pandas as pd
import numpy as np
from utils import time_parser


TIMESTEPS = args.timesteps
NUM_FEATURE = 22
X = None
Y = None
# for m in range(5):
#     name = 'MSFT'
#     day_index = m + 1
#     path_lob_train = './LOBSTER/%s_orderbook_part_%s.csv'%(name,day_index)
#     path_msb_train = './LOBSTER/%s_message_part_%s.csv'%(name,day_index)
#     data_lob, transaction = time_parser(path_lob_train, path_msb_train)
#     mid_prices = (data_lob.iloc[:,0] + data_lob.iloc[:,2])/2
#
#     data_lob = data_lob.iloc[:,[0,1,2,3]]
#     num_samples = len(data_lob) - TIMESTEPS
#     ROLLING = args.rolling
#     mid_prices_rolling = mid_prices.rolling(ROLLING).mean().shift(-(ROLLING-1)).dropna()
#     for i in range(num_samples-ROLLING+1):
#         data = pd.DataFrame(np.zeros(shape = (50,22)))
#         data = data_lob.iloc[i:(i+TIMESTEPS),:4].copy()
#         data.iloc[:,[0,2]] = (data.iloc[:,[0,2]] - data.iloc[-1,[0,2]])/100
#         ask_encoder = np.zeros(shape=(args.timesteps,11))
#         bid_encoder = np.zeros(shape=(args.timesteps,11))
#         for j in range(args.timesteps):
#             if 5+int(data.iloc[-(j+1),0])>=0 and 5+int(data.iloc[-(j+1),0])<=10:
#                 ask_encoder[-(j+1),5+int(data.iloc[-(j+1),0])] = data.iloc[-(j+1),1]
#             if 5+int(data.iloc[-(j+1),2])>=0 and 5+int(data.iloc[-(j+1),2])<=10:
#                 bid_encoder[-(j+1),5+int(data.iloc[-(j+1),2])] = data.iloc[-(j+1),3]
#         data_encoded = np.hstack((ask_encoder,bid_encoder))
#         if X is None:
#             X = data_encoded[None,:]
#         else:
#             X = np.concatenate((X,data_encoded[None,:]),axis=0)
#         if mid_prices_rolling[i+TIMESTEPS] - mid_prices_rolling[i+TIMESTEPS-1] > 0:
#             y=2
#         elif mid_prices_rolling[i+TIMESTEPS] - mid_prices_rolling[i+TIMESTEPS-1] < 0:
#             y=0
#         else:
#             y=1
#         if Y is None:
#             Y = np.array([y])[None,:]
#         else:
#             Y = np.concatenate((Y,np.array([y])[None,:]),axis=0)
#     if m == 2:
#         np.save('./sparse_encoded_trend_prediction_X_train.npy',X)
#         np.save('./sparse_encoded_trend_prediction_Y_train.npy',Y)
#         X = None
#         Y = None
#     if m == 3:
#         np.save('./sparse_encoded_trend_prediction_X_val.npy',X)
#         np.save('./sparse_encoded_trend_prediction_Y_val.npy',Y)
#         X = None
#         Y = None
#     if m == 4:
#         np.save('./sparse_encoded_trend_prediction_X_test.npy',X)
#         np.save('./sparse_encoded_trend_prediction_Y_test.npy',Y)


NUM_FEATURE = 2
# for m in range(5):
#     name = 'MSFT'
#     day_index = m + 1
#     path_lob_train = './LOBSTER/%s_orderbook_part_%s.csv'%(name,day_index)
#     path_msb_train = './LOBSTER/%s_message_part_%s.csv'%(name,day_index)
#     data_lob, transaction = time_parser(path_lob_train, path_msb_train)
#     mid_prices = (data_lob.iloc[:,0] + data_lob.iloc[:,2])/2
#
#     data_lob = data_lob.iloc[:,[1,3]]
#     num_samples = len(data_lob) - TIMESTEPS
#     ROLLING = args.rolling
#     mid_prices_rolling = mid_prices.rolling(ROLLING).mean().shift(-(ROLLING-1)).dropna()
#     for i in range(num_samples-ROLLING+1):
#         data = pd.DataFrame(np.zeros(shape = (50,2)))
#         data = np.array(data_lob.iloc[i:(i+TIMESTEPS),:2].copy())
#         if X is None:
#             X = data[None,:]
#         else:
#             X = np.concatenate((X,data[None,:]),axis=0)
#         if mid_prices_rolling[i+TIMESTEPS] - mid_prices_rolling[i+TIMESTEPS-1] > 0:
#             y=2
#         elif mid_prices_rolling[i+TIMESTEPS] - mid_prices_rolling[i+TIMESTEPS-1] < 0:
#             y=0
#         else:
#             y=1
#         if Y is None:
#             Y = np.array([y])[None,:]
#         else:
#             Y = np.concatenate((Y,np.array([y])[None,:]),axis=0)
#     if m == 2:
#         np.save('./explicit_noprice_encoded_trend_prediction_X_train.npy',X)
#         np.save('./explicit_noprice_encoded_trend_prediction_Y_train.npy',Y)
#         X = None
#         Y = None
#     if m == 3:
#         np.save('./explicit_noprice_encoded_trend_prediction_X_val.npy',X)
#         np.save('./explicit_noprice_encoded_trend_prediction_Y_val.npy',Y)
#         X = None
#         Y = None
#     if m == 4:
#         np.save('./explicit_noprice_encoded_trend_prediction_X_test.npy',X)
#         np.save('./explicit_noprice_encoded_trend_prediction_Y_test.npy',Y)