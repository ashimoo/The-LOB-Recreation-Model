import argparse
import os
import pandas as pd
import numpy as np
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random
from tensorflow.keras.utils import to_categorical
import sklearn
from tqdm import tqdm
import torch.nn.functional as F

# Argument Parsing
args = argparse.ArgumentParser('predict')
args.add_argument('--seed', type=int, default=0, help="random seed")
args.add_argument('--rolling', type=int, default=5, help="num of rolling")
args.add_argument('--timesteps', type=int, default=100)
args = args.parse_args()


# Data Preparation
def model_parser(data_lob):
    mid_prices = (data_lob.iloc[:, 0].copy() + data_lob.iloc[:, 2].copy()) / 2

    data_lob.iloc[:,list(range(1,NUM_FEATURE+1,2))] = data_lob.iloc[:,list(range(1,NUM_FEATURE+1,2))].clip(upper=data_lob.iloc[:int(len(data_lob)*0.9),list(range(1,NUM_FEATURE+1,2))].quantile(0.95),axis=1)

    # scaler = sklearn.preprocessing.MinMaxScaler()
    # scaler.fit(data_lob.iloc[:int(len(data_lob)*0.9)])
    # data_lob = pd.DataFrame(scaler.transform(data_lob))

    if params[2] == 'taq':
        data_lob.iloc[:, list(range(5, NUM_FEATURE + 1, 2))] = 0

    num_samples = len(data_lob) - TIMESTEPS
    X = np.zeros(shape=(num_samples, TIMESTEPS, int(NUM_FEATURE)))
    Y = np.zeros(shape=(num_samples, 1))
    ROLLING = args.rolling
    mid_prices_rolling = mid_prices.rolling(ROLLING).mean().shift(-(ROLLING - 1)).dropna()
    for i in range(num_samples - ROLLING + 1):
        target = data_lob.iloc[i:(i + TIMESTEPS), :].copy()
        X[i, :, :] = target.values
        if mid_prices_rolling[i + TIMESTEPS] - mid_prices[i + TIMESTEPS - 1] > 0:
            Y[i, :] = 2
        elif mid_prices_rolling[i + TIMESTEPS] - mid_prices[i + TIMESTEPS - 1] < 0:
            Y[i, :] = 0
        else:
            Y[i, :] = 1
    X = X.reshape(num_samples, TIMESTEPS, int(NUM_FEATURE), 1)
    Y = to_categorical(Y[:, 0])
    return X, Y


# Define the Model
class deeplob(nn.Module):
    def __init__(self, y_len):
        super().__init__()
        self.y_len = y_len

        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 5)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x, device='cuda'):
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), 64).to(device)
        c0 = torch.zeros(1, x.size(0), 64).to(device)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        forecast_y = torch.softmax(x, dim=1)

        return forecast_y

dataset = 'MSFT'

for params in [[20,'real_lob','lob']]:
    TIMESTEPS = args.timesteps
    NUM_FEATURE = params[0]

    data = pd.read_csv('./fake_lob/{}_{}.csv'.format(params[1],dataset),index_col=0)
    data.reset_index(drop=True,inplace=True)
    data = data.iloc[:,:NUM_FEATURE]


    # Load Data
    X_data, Y_data = model_parser(data)
    X_train, X_val, X_test = X_data[:int(0.8*len(X_data)),:], X_data[int(0.8*len(X_data)):int(0.9*len(X_data)),:], X_data[int(0.9*len(X_data)):,:]
    Y_train, Y_val, Y_test = Y_data[:int(0.8*len(X_data)),:], Y_data[int(0.8*len(X_data)):int(0.9*len(X_data)),:], Y_data[int(0.9*len(X_data)):,:]

    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32).permute(0,3,1,2).cuda(), torch.tensor(np.argmax(Y_train, axis=1), dtype=torch.long).cuda())
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32).permute(0,3,1,2).cuda(), torch.tensor(np.argmax(Y_val, axis=1), dtype=torch.long).cuda())
    val_loader = DataLoader(val_data, batch_size=64)

    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32).permute(0,3,1,2).cuda(), torch.tensor(np.argmax(Y_test, axis=1), dtype=torch.long).cuda())
    test_loader = DataLoader(test_data, batch_size=64)

    test_acc = []
    for r in range(1):
        # Create the model, optimizer, and loss function
        model = deeplob(100).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        # Training Loop
        acc = 0
        for epoch in tqdm(range(200)):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                correct = 0
                for data, target in val_loader:
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

                if correct / len(val_loader.dataset)>acc:
                    torch.save({'state_dict': model.state_dict(), }, './model.ckpt')
                    acc = correct / len(val_loader.dataset)

        # Evaluation Loop
        model.load_state_dict(torch.load('./model.ckpt')['state_dict'])
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # print(f"Test loss: {test_loss}, Accuracy: {correct/len(test_loader.dataset)}")
        test_acc.append(correct/len(test_loader.dataset))
    print(np.mean(test_acc))