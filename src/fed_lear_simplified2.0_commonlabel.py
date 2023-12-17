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

IS_FEDERATED_ACTIVE = 'Y'

y_encoding = {
    "Downstairs": 0,
    "Jogging": 1,
    "Sitting": 2,
    "Standing": 3,
    "Upstairs": 4,
    "Walking": 5
}
def load_data():
    data = np.loadtxt('dataset.txt', delimiter=',', dtype=object)
    data = data[:, [0, 1, 3, 4, 5]]
    data_pd = pd.DataFrame(data, columns=['client', 'activity', 'x_acc', 'y_acc', 'z_acc'])
    data_pd['x_acc'] = data_pd['x_acc'].astype(float)
    data_pd['y_acc'] = data_pd['y_acc'].astype(float)
    data_pd['z_acc'] = data_pd['z_acc'].astype(float)
    data_pd['client'] = data_pd['client'].astype(int)
    data_pd['activity'] = data_pd['activity'].map(y_encoding)
    data_pd['activity'] = data_pd['activity'].astype(int)
    data = data_pd.to_numpy()
    return data

def load_data_with_timestamp():
    data = np.loadtxt('dataset.txt', delimiter=',', dtype=object)
    data = data[:, [0, 1, 2, 3, 4, 5]]
    data_pd = pd.DataFrame(data, columns=['client', 'activity', 'timestamp', 'x_acc', 'y_acc', 'z_acc'])
    data_pd['x_acc'] = data_pd['x_acc'].astype(float)
    data_pd['y_acc'] = data_pd['y_acc'].astype(float)
    data_pd['z_acc'] = data_pd['z_acc'].astype(float)
    data_pd['client'] = data_pd['client'].astype(int)
    data_pd['activity'] = data_pd['activity'].map(y_encoding)
    data_pd['activity'] = data_pd['activity'].astype(int)
    data = data_pd.to_numpy()
    return data

DATA_TIMESTAMP = load_data_with_timestamp()
DATA = load_data()

def load(conf_file):
    with open(conf_file) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        return settings

def splitTrainValDataset(x_tensor, y_tensor):
    ratio = .7
    dataset = TensorDataset(x_tensor, y_tensor)
    print(f"number of rows: {len(dataset)}" )
    n_total = len(dataset)
    n_train = int(n_total * ratio)
    n_val = n_total - n_train
    train_data, val_data = random_split(dataset, [n_train, n_val])
    y_tensor_train = y_tensor[train_data.indices]
    y_val_train = y_tensor[val_data.indices]
    classes, counts = y_tensor_train.unique(return_counts=True)
    print(f"number of activities: {len(classes)}")
    c = counts.numpy()
    print(f"number of row for each activity: {c/c.sum()}")
    #classes, counts = y_val_train.unique(return_counts=True)
    #print(classes, counts)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=100,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=1000
    )
    return train_loader, val_loader, val_data


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
            self.total_epochs += 1
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)
            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        if self.val_loader:
            plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

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

def load_client_data(client):
    data_client = DATA[DATA[:, 0] == client]
    x = data_client[:, [2, 3, 4]].astype(float)
    y = data_client[:, 1]
    x_tensor = torch.as_tensor(x).float()
    y_tensor = torch.as_tensor(y.reshape(-1, 1)).float()
    return x_tensor, y_tensor

#logging.basicConfig(level=logging.DEBUG)

class Setup:
    def __init__(self, conf_file):
        self.conf_file = conf_file
        self.settings = load(self.conf_file)
        self.data_path = self.settings['setup']['data_path']
        self.n_clients = self.settings['setup']['n_clients']
        self.learning_rate = self.settings['setup']['learning_rate']
        self.num_of_epochs = self.settings['setup']['num_of_epochs']
        self.batch_size = self.settings['setup']['batch_size']
        self.saving_dir = self.settings['setup']['save_dir']
        self.to_save = self.settings['setup']['to_save']
        self.federated_runs = self.settings['setup']['federated_runs']
        self.saved = False

        if "saved" not in self.settings.keys():
            self.start_time = datetime.now()
        else:
            self.saved = True
            self.start_time = datetime.strptime(
                self.settings['saved']['timestamp'], '%Y%m%d%H%M')

        timestamp = self.start_time.strftime("%Y%m%d%H%M")
        self.path = os.path.join(self.saving_dir, timestamp)
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self.list_of_clients = []

        logging.debug("Setup: creating clients with path %s",
                      self.path)
        self.create_clients()

        logging.debug("Setup: creating server with path %s",
                      self.path)
        self.server = Server(self.list_of_clients, self.path, self.learning_rate, self.num_of_epochs)

    def run(self):
        for i in range(self.federated_runs):
            logging.info(
                "Setup: starting run of the federated learning number %s", (i + 1))
            print(f"Setup: starting run of the federated learning number {i+1}")
            if not i == 0:
                self.server.update_server()
                print(f"Setup: train server")
                self.server.architecture.train(self.num_of_epochs)
            for c in self.server.list_of_clients:
                if IS_FEDERATED_ACTIVE == 'Y':
                    c.load_server_weights()
                    print(f"Setup: train client_{c.identifier}")
                    c.architecture.train(self.num_of_epochs)

    def create_clients(self):
        for i in range(1, self.n_clients):
            print("client: " + str(i))
            c = Client(i, self.path, self.learning_rate, self.num_of_epochs, self.batch_size)
            self.list_of_clients.append(c)

    def save_models(self):
        for c in self.list_of_clients:
            c.save_model(self.path)

    def save(self):
        timestamp = self.start_time.strftime("%Y%m%d%H%M")
        self.path = os.path.join(self.saving_dir, timestamp)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.save_models()
        self.settings['saved'] = {"timestamp": timestamp}
        with open(os.path.join(self.path, 'setup.yaml'), 'w') as fout:
            yaml.dump(self.settings, fout)


class Server:
    def __init__(self, list_of_clients, path, learning_rate, num_of_epochs):
        self.learning_rate = learning_rate
        self.list_of_clients = list_of_clients
        self.path = path
        self.num_of_epochs = num_of_epochs
        self.network = nn.Sequential()
        self.network.add_module('linear_1', nn.Linear(3, 64))
        self.network.add_module("relu_1", nn.ReLU())
        self.network.add_module('linear_2', nn.Linear(64, 128))
        self.network.add_module("relu_2", nn.ReLU())
        self.network.add_module('linear_3', nn.Linear(128, 64))
        self.network.add_module("relu_3", nn.ReLU())
        self.network.add_module('linear_4', nn.Linear(64, 6))
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.architecture = Architecture(self.network, self.loss_fn, self.optimizer)

        self.save_server()

    def update_server(self):
        checkpoint_server = torch.load(os.path.join(self.path, "model_server.pth"))
        models_array = []
        for c in self.list_of_clients:
            checkpoint_client = torch.load(os.path.join(c.path, "model_{}.pth".format(c.identifier)))
            models_array.append(checkpoint_client['model_state_dict'])
        for key in list(models_array[0].keys()):
            attribute_array = [i[key] for i in models_array]
            attribute_sum = sum(attribute_array)/len(attribute_array)
            checkpoint_server['model_state_dict'][key] = attribute_sum
        self.architecture.model.load_state_dict(checkpoint_server['model_state_dict'])
        self.save_server()

    def save_server(self):
        self.architecture.save_checkpoint(os.path.join(self.path, "model_server.pth"))


class Client:
    def __init__(self, identifier, path, learning_rate,
                 num_of_epochs, batch_size):
        self.identifier = identifier
        self.path = path
        self.learning_rate = learning_rate
        self.num_of_epochs = num_of_epochs
        self.batch_size = batch_size
        self.x, self.y = load_client_data(self.identifier)
        self.train_dataset_loader, self.val_dataset_loader, self.val_data = splitTrainValDataset(self.x, self.y)
        self.network = nn.Sequential()
        self.network.add_module('linear_1', nn.Linear(3, 64))
        self.network.add_module("relu_1", nn.ReLU())
        self.network.add_module('linear_2', nn.Linear(64, 128))
        self.network.add_module("relu_2", nn.ReLU())
        self.network.add_module('linear_3', nn.Linear(128, 64))
        self.network.add_module("relu_3", nn.ReLU())
        self.network.add_module('linear_4', nn.Linear(64, 6))
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.architecture = Architecture(self.network, self.loss_fn, self.optimizer)
        self.architecture.set_loaders(self.train_dataset_loader, self.val_dataset_loader)

        self.save_model()


    def load_server_weights(self):
        with torch.no_grad():
            checkpoint_server = torch.load(os.path.join(self.path, "model_server.pth"))
            checkpoint_client = torch.load(os.path.join(self.path, "model_{}.pth".format(self.identifier)))
            for key in checkpoint_server['model_state_dict']:
                checkpoint_client['model_state_dict'][key] = checkpoint_server['model_state_dict'][key]
            self.architecture.model.load_state_dict(checkpoint_client['model_state_dict'])
            self.save_model()


    def save_model(self):
        self.architecture.save_checkpoint(os.path.join(self.path, "model_{}.pth".format(self.identifier)))


if __name__ == '__main__':
    conf_file = 'conf_file.yml'
    setup = Setup(conf_file)
    #logging.basicConfig(level=logging.DEBUG)
    logging.debug("Running with configuration file")

    setup.run()
    for c in setup.server.list_of_clients:
        #fig = c.architecture.plot_losses()
        cm = c.architecture.correct(c.val_data[:][0], c.val_data[:][1])
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        #plt.show()
        accuracy = np.trace(cm) / np.sum(cm)
        print(accuracy)
    #if setup.to_save:
     #   setup.save()