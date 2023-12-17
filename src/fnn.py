import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, \
  TensorDataset
from torch.nn.utils import rnn as rnn_utils
import pandas as pd
from datetime import datetime
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.metrics import accuracy_score


class Architecture(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.train_loader = None
        self.val_loader = None
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0
        self.train_step = self._make_train_step()
        self.val_step = self._make_val_step()

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _make_train_step(self):
        def perform_train_step(x, y):
            self.model.train()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()

        return perform_train_step

    def _make_val_step(self):
        def perform_val_step(x,y):
            self.model.eval()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            return loss.item()

        return perform_val_step

    def model_state(self):
        return self.model.state_dict()

    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader
            step = self.val_step
        else:
            data_loader = self.train_loader
            step = self.train_step

        if data_loader is None:
            return None

        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)

        return loss

    def set_seed(self, seed = 42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self, n_epochs, seed=42):
        self.set_seed(seed)
        for epoch in range(n_epochs):
            print(f"Starting epoch: {self.total_epochs + 1}")
            self.total_epochs += 1
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)
            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)


for client in range(1, 37):
    data = torch.load(f'..\data\\fnn\dataset_fnn_{client}.pt')
    train_ratio = 0.8
    train_size = int(train_ratio * len(data))
    test_size = len(data) - train_size
    indices = torch.randperm(len(data)).tolist()
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(data, train_indices)
    test_dataset = Subset(data, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    network = nn.Sequential()
    network.add_module('linear_1', nn.Linear(9, 64))
    network.add_module("relu_1", nn.ReLU())
    network.add_module('linear_2', nn.Linear(64, 128))
    network.add_module("relu_2", nn.ReLU())
    network.add_module('linear_3', nn.Linear(128, 64))
    network.add_module("relu_3", nn.ReLU())
    network.add_module('linear_4', nn.Linear(64, 6))
    optimizer = optim.SGD(network.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    architecture = Architecture(network, loss_fn, optimizer)
    architecture.set_loaders(train_loader, test_loader)
    architecture.train(15)

    architecture.model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            outputs = architecture.model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
            actual_labels.extend(labels.tolist())


    accuracy = accuracy_score(actual_labels, predictions)
    print(f'Accuracy client_{client}: {accuracy:.4f}')



