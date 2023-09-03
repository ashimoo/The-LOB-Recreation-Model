from __future__ import division
import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from sklearn.linear_model import LinearRegression
import torch

import scipy

l1_train_list = []
l1_val_list = []
l1_val_list_itr = []
l1_test_list = []
y_pred = None
y_real = None
side = 'ask'
stock = 'JPM'
model_name = 'LSVR'
r_list = []
loss_list = []

raw = torch.load('./parsed_data_/data_{}_{}_ws005_train_3days_1interval_timezscore_together_4exactlabel_sparse_100_3_new.pt'.format(side,stock))
x_train = np.vstack([item[2].reshape(-1).cpu().numpy() for item in raw])
y_train = np.vstack([item[4].cpu().numpy() for item in raw])
raw = torch.load('./parsed_data_/data_{}_{}_ws005_val_1day_1interval_timezscore_together_4exactlabel_sparse_100_3_new.pt'.format(side,stock))
x_val = np.vstack([item[2].reshape(-1).cpu().numpy() for item in raw])
y_val = np.vstack([item[4].cpu().numpy() for item in raw])
raw = torch.load('./parsed_data_/data_{}_{}_ws005_test_1day_1interval_timezscore_together_4exactlabel_sparse_100_new.pt'.format(side,stock))
x_test = np.vstack([item[2].reshape(-1).cpu().numpy() for item in raw])
y_test = np.vstack([item[4].cpu().numpy() for item in raw])

pca = PCA(n_components=60,random_state=0)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_val = pca.transform(x_val)
x_test = pca.transform(x_test)

for rounds in range(5):
    if model_name == 'xgboost':
        for i in range(4):
            model = XGBRegressor(random_state=rounds)
            model.fit(x_train,y_train[:,i],eval_set=[(x_val,y_val[:,i])],eval_metric='mae')
            y_test_hat = model.predict(x_test)
            y_val_hat = model.predict(x_val)
            y_train_hat = model.predict(x_train)
            l1_test_list.append(mean_absolute_error(y_test[:,i], y_test_hat))
            l1_val_list.append(mean_absolute_error(y_val[:,i], y_val_hat))
            l1_train_list.append(mean_absolute_error(y_train[:,i], y_train_hat))
            if y_pred is None:
                y_pred = model.predict(x_test)
                y_real = y_test[:,i]
            else:
                y_pred = np.concatenate((y_pred,model.predict(x_test)))
                y_real = np.concatenate((y_real,y_test[:,i]))
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_pred, y_real)
        print('R_squared value for ask side prediction is %f'%r_value**2)
        print(sum(l1_train_list)/len(l1_train_list),sum(l1_val_list)/len(l1_val_list),sum(l1_test_list)/len(l1_test_list))
        loss_list.append(sum(l1_test_list)/len(l1_test_list))
        r_list.append(r_value**2)

    if model_name == 'LSVR' or model_name == 'RR':
        for i in range(4):
            if model_name == 'LSVR':
                model = LinearSVR()
            else:
                model = Ridge()

            model.fit(x_train, y_train[:, i])
            y_train_hat = model.predict(x_train)
            y_val_hat = model.predict(x_val)
            y_test_hat = model.predict(x_test)
            l1_train_list.append(mean_absolute_error(y_train[:, i], y_train_hat))
            l1_test_list.append(mean_absolute_error(y_test[:, i], y_test_hat))
            l1_val_list.append(mean_absolute_error(y_val[:, i], y_val_hat))
            if y_pred is None:
                y_pred = model.predict(x_test)
                y_real = y_test[:,i]
            else:
                y_pred = np.concatenate((y_pred,model.predict(x_test)))
                y_real = np.concatenate((y_real,y_test[:,i]))
            print('done')
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_pred, y_real)
        print('R_squared value for ask side prediction is %f'%r_value**2)
        print(sum(l1_train_list)/len(l1_train_list),sum(l1_val_list)/len(l1_val_list),sum(l1_test_list)/len(l1_test_list))
        loss_list.append(sum(l1_test_list)/len(l1_test_list))
        r_list.append(r_value**2)

    if model_name == 'SLFN':
        model = Sequential([
            Dense(128, input_dim=60),
            Activation('tanh'),
            Dense(4)]
        )

        optim = optimizers.RMSprop(lr=1e-4)
        model.compile(optimizer=optim, loss='mean_absolute_error')
        model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
        callbacks_list = [model_checkpoint]
        model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), batch_size=512, shuffle=True,
                  callbacks=callbacks_list)
        model.load_weights('best_model.h5')
        print(model.evaluate(x_test, y_test))

        y_pred = model.predict(x_test).reshape(-1)
        y_real = y_test.reshape(-1)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_pred, y_real)
        print('R_squared value for ask side prediction is %f' % r_value ** 2)
        loss_list.append(model.evaluate(x_test, y_test))
        r_list.append(r_value**2)
print(np.mean(loss_list),np.mean(r_list))
