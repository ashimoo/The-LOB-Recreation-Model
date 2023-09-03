
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import random
import sklearn
from utils import time_parser
import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.utils import to_categorical
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

for seed in range(5):

    torch.manual_seed(seed)
    np.random.seed(seed)

    mode = 'explicit_noprice'
    stan = 'zscore'

    X_train = np.load('./parsed_data_/{}_encoded_trend_prediction_X_train.npy'.format(mode))
    X_val = np.load('./parsed_data_/{}_encoded_trend_prediction_X_val.npy'.format(mode))
    X_test = np.load('./parsed_data_/{}_encoded_trend_prediction_X_test.npy'.format(mode))
    Y_train = to_categorical(np.load('./parsed_data_/{}_encoded_trend_prediction_Y_train.npy'.format(mode)))
    Y_val = to_categorical(np.load('./parsed_data_/{}_encoded_trend_prediction_Y_val.npy'.format(mode)))
    Y_test = to_categorical(np.load('./parsed_data_/{}_encoded_trend_prediction_Y_test.npy'.format(mode)))

    X = np.concatenate((X_train,X_val),axis=0)
    if stan == 'zscore':
        scaler=sklearn.preprocessing.StandardScaler().fit(X.reshape(len(Y_train)+len(Y_val),-1))
    elif stan == 'minmax':
        scaler = sklearn.preprocessing.MinMaxScaler().fit(X.reshape(len(Y_train) + len(Y_val), -1))

    X_train=scaler.transform(X_train.reshape(len(Y_train),-1)).reshape(len(Y_train),50,-1)
    X_val=scaler.transform(X_val.reshape(len(Y_val),-1)).reshape(len(Y_val),50,-1)
    X_test=scaler.transform(X_test.reshape(len(Y_test),-1)).reshape(len(Y_test),50,-1)

    TIMESTEPS = 50
    if mode == 'explicit':
        NUM_FEATURES = 4
    elif mode == 'explicit_noprice':
        NUM_FEATURES = 2
    elif mode == 'sparse':
        NUM_FEATURES = 22

    # Define the PyTorch model
    class MyModel(nn.Module):
        def __init__(self, mode):
            super(MyModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2))
            self.conv2 = nn.Conv2d(16, 16, kernel_size=(1, 2), stride=(1, 2))
            self.dense1 = nn.Linear(16, 16)
            self.dense2 = nn.Linear(22, 16)
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.gru = nn.GRU(16, 32, batch_first=True)
            self.dense3 = nn.Linear(32, 3)
            self.mode = mode

        def forward(self, x):
            if mode == 'explicit':
                x = self.conv1(x)
                x = self.leaky_relu(x)
                x = self.conv2(x)
                x = self.leaky_relu(x)
                x = x.squeeze(-1).permute(0,2,1)
                x = self.dense1(x)
                x = self.relu(x)
            elif mode == 'explicit_noprice':
                x = self.conv1(x)
                x = self.leaky_relu(x)
                x = x.squeeze(-1).permute(0,2,1)
                x = self.dense1(x)
                x = self.relu(x)
            elif mode == 'sparse':
                x = self.dense2(x)
                x = self.relu(x).squeeze(1)

            x, _ = self.gru(x)
            x = self.dense3(x[:, -1, :])  # Take the last output from GRU as the final output
            return x

    # Create an instance of the model
    model = MyModel(mode = mode)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    best_val_acc = 0

    # Convert data to PyTorch tensors
    X_train_torch = torch.tensor(X_train[:, :, :, None], dtype=torch.float32).permute(0,3,1,2)
    Y_train_torch = torch.tensor(Y_train, dtype=torch.float)
    X_val_torch = torch.tensor(X_val[:, :, :, None], dtype=torch.float32).permute(0,3,1,2)
    Y_val_torch = torch.tensor(Y_val, dtype=torch.float)

    train_dataset = CustomDataset(X_train_torch, Y_train_torch)
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_torch)
            _, val_preds = torch.max(val_outputs, 1)
            val_acc = np.sum((val_preds.detach().numpy() == np.argmax(Y_val, axis=1))) / len(Y_val)

            if val_acc > best_val_acc:
                torch.save({'state_dict': model.state_dict(), }, './model.ckpt')
                best_val_acc = val_acc

    model.load_state_dict(torch.load('./model.ckpt')['state_dict'])
    # Evaluation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_torch)
        _, val_preds = torch.max(val_outputs, 1)
        test_outputs = model(torch.tensor(X_test[:, :, :, None], dtype=torch.float32).permute(0,3,1,2))
        _, test_preds = torch.max(test_outputs, 1)

    val_accuracy = np.sum((val_preds.detach().numpy() == np.argmax(Y_val,axis=1))) / len(Y_val)
    test_accuracy = np.sum((test_preds.detach().numpy() == np.argmax(Y_test,axis=1))) / len(Y_test)

    print("Val Accuracy: {:.2f}%".format(val_accuracy * 100))
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))