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
import seaborn as sns
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Subset

plt.ion()

from datetime import datetime
import yaml
import os
import logging
from torch.utils.data import DataLoader, TensorDataset, random_split
from utils.architecture_class import Architecture

IS_FEDERATED_ACTIVE = 'Y'
PLOT = 'Y'

def load(conf_file):
    with open(conf_file) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        return settings


def load_data_client(client, network_type):
    if network_type == 'fnn':
        data = torch.load(f'..\data\\fnn\dataset_fnn_{client}.pt')
    elif network_type == 'lstm':
        data = torch.load(f'..\data\\fnn\dataset_lstm_{client}.pt')
    else:
        print('no network selected')
        data = 0
    train_ratio = .8
    train_size = int(train_ratio * len(data))
    test_size = len(data) - train_size
    indices = torch.randperm(len(data)).tolist()
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_dataset = Subset(data, train_indices)
    test_dataset = Subset(data, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader


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
        self.network_type = self.settings['setup']['network_type']
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
                #HOW IS IT POSSIBLE TO TRAIN THE SERVER??
                #print(f"Setup: train server")
                #self.server.architecture.train(self.num_of_epochs)
            for c in self.server.list_of_clients:
                if IS_FEDERATED_ACTIVE == 'Y':
                    c.load_server_weights()
                    print(f"Setup: train client_{c.identifier}")
                    c.architecture.train(self.num_of_epochs)
                    c.architecture.get_accuracy()
                    if PLOT == 'Y':
                        c.architecture.plot_losses()
    def create_clients(self):
        for i in range(1, self.n_clients):
            print("created client: " + str(i))
            c = Client(i, self.path, self.learning_rate, self.num_of_epochs, self.batch_size, self.network_type)
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
        self.network.add_module('linear_1', nn.Linear(9, 64))
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
                 num_of_epochs, batch_size, network_type):
        self.identifier = identifier
        self.path = path
        self.learning_rate = learning_rate
        self.num_of_epochs = num_of_epochs
        self.batch_size = batch_size
        self.network_type = network_type
        self.train_loader, self.test_loader = load_data_client(self.identifier, self.network_type)
        self.network = nn.Sequential()
        self.network.add_module('linear_1', nn.Linear(9, 64))
        self.network.add_module("relu_1", nn.ReLU())
        self.network.add_module('linear_2', nn.Linear(64, 128))
        self.network.add_module("relu_2", nn.ReLU())
        self.network.add_module('linear_3', nn.Linear(128, 64))
        self.network.add_module("relu_3", nn.ReLU())
        self.network.add_module('linear_4', nn.Linear(64, 6))
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.architecture = Architecture(self.network, self.loss_fn, self.optimizer)
        self.architecture.set_loaders(self.train_loader, self.test_loader)

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