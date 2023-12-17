import numpy as np
from sklearn.utils import shuffle
from keras.datasets import mnist
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import argparse
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

plt.ion()


import random
import math
from datetime import datetime
import torch.nn.functional as F
import yaml
import os
from operator import itemgetter
import logging
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score

class Architecture(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.train_loader = None
        self.test_loader = None
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0
        self.train_step = self._make_train_step()
        self.val_step = self._make_val_step()

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def set_loaders(self, train_loader, test_loader=None):
        self.train_loader = train_loader
        self.test_loader = test_loader

    def _make_train_step(self):
        def perform_train_step(x, y):
            self.model.train()
            yhat = self.model(x)
            #yhat = torch.argmax(nn.Softmax(dim=-1)(yhat), dim=1, keepdim=True).float()
            loss = self.loss_fn(yhat, y.squeeze().long())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()

        return perform_train_step

    def _make_val_step(self):
        def perform_val_step(x,y):
            self.model.eval()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y.squeeze().long())
            return loss.item()

        return perform_val_step

    def model_state(self):
        return self.model.state_dict()

    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.test_loader
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

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        if self.test_loader:
            plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            #'loss': self.losses,
            #'val_loss': self.val_losses
        }

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )

        #self.total_epochs = checkpoint['epoch']
        #self.losses = checkpoint['loss']
        #self.val_losses = checkpoint['val_loss']

        self.model.train()

    def correct(self, x, y, threshold=.5):
        self.model.eval()
        yhat = self.model(x.to(self.device))
        y = y.to(self.device)
        self.model.train()


        n_samples, n_dims = yhat.shape
        _, predicted = torch.max(yhat, 1)
        cm = confusion_matrix(y, predicted)
        return cm

    def get_accuracy(self):
        self.model.eval()
        predictions = []
        actual_labels = []

        with torch.no_grad():
            for sequences, labels in self.test_loader:
                outputs = self.model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.tolist())
                actual_labels.extend(labels.tolist())

        accuracy = accuracy_score(actual_labels, predictions)
        print(f'Accuracy: {accuracy:.4f}')