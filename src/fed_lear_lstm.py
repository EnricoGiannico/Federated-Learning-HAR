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
from sklearn.metrics import accuracy_score


IS_FEDERATED_ACTIVE = 'Y'
#PLOT = 'N'
input_size = 3
hidden_size = 50
num_classes = 6


def train(num_epoch, train_loader, test_loader, optimizer, model, criterion):
    train_loss = []
    test_loss = []
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        for sequences, labels in train_loader:
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        average_loss = epoch_loss / len(train_loader)
        train_loss.append(average_loss)
        print(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {average_loss:.4f}')
        ####test loss
        t_epoch_loss = 0.0

        for sequences, labels in test_loader:
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_epoch_loss += loss.item()

        t_average_loss = t_epoch_loss / len(test_loader)
        test_loss.append(t_average_loss)
    return train_loss, test_loss


def get_accuracy(model, test_loader):
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
            actual_labels.extend(labels.tolist())

    accuracy = accuracy_score(actual_labels, predictions)
    print(f'Accuracy: {accuracy:.4f}')

def plot_losses(train_loss, test_loss):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(train_loss, label='Training Loss', c='b')
    plt.plot(test_loss, label='Test Loss', c='r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]

        out = self.fc(out)
        return out

    def get_state(self):
        lstm_dict = self.lstm.state_dict()
        fc_dict = self.fc.state_dict()
        dict_state = {
            'lstm': lstm_dict,
            'fc': fc_dict
        }
        return dict_state


    def save_state(self, filename):
        state = self.get_state()
        torch.save(state, filename)

    def load_state(self, checkpoint):
        self.lstm.load_state_dict(checkpoint['lstm'])
        self.fc.load_state_dict(checkpoint['fc'])

    def load_state_file(self, filename):
        checkpoint = torch.load(filename)
        self.lstm.load_state_dict(checkpoint['lstm'])
        self.fc.load_state_dict(checkpoint['fc'])




def load(conf_file):
    with open(conf_file) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        return settings


def load_data_client(client, network_type):
    data = torch.load(f'..\data\\lstm\dataset_lstm_{client}.pt')
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
            for c in self.server.list_of_clients:
                if IS_FEDERATED_ACTIVE == 'Y':
                    c.load_server_weights()
                print(f"Setup: train client_{c.identifier}")
                train_loss, test_loss = train(self.num_of_epochs, c.train_loader, c.test_loader, c.optimizer, c.model, c.criterion)
                c.add_loss(train_loss, test_loss)
                    #if PLOT == 'Y':
                     #   c.architecture.plot_losses()
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
        self.model = LSTMClassifier(input_size, hidden_size, num_classes)

        self.save_server()

    def update_server(self):
        checkpoint_server = torch.load(os.path.join(self.path, "model_server.pth"))
        models_lstm_array = []
        models_fc_array = []
        for c in self.list_of_clients:
            checkpoint_client = torch.load(os.path.join(c.path, "model_{}.pth".format(c.identifier)))
            models_lstm_array.append(checkpoint_client['lstm'])
            models_fc_array.append(checkpoint_client['fc'])
        for key1 in list(models_fc_array[0].keys()):
            attribute_array = [i[key1] for i in models_fc_array]
            attribute_sum = sum(attribute_array)/len(attribute_array)
            checkpoint_server['fc'][key1] = attribute_sum
        for key2 in list(models_lstm_array[0].keys()):
            attribute_array = [i[key2] for i in models_lstm_array]
            attribute_sum = sum(attribute_array) / len(attribute_array)
            checkpoint_server['lstm'][key2] = attribute_sum

        self.model.load_state(checkpoint_server)
        self.save_server()

    def save_server(self):
        self.model.save_state(os.path.join(self.path, "model_server.pth"))


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
        self.model = LSTMClassifier(input_size, hidden_size, num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loss = []
        self.test_loss = []


        self.save_model()


    def load_server_weights(self):
        self.model.load_state_file(os.path.join(self.path, "model_server.pth"))
        self.save_model()

    def add_loss(self, train_loss, test_loss):
        self.train_loss += train_loss
        self.test_loss += test_loss


    def save_model(self):
        self.model.save_state(os.path.join(self.path, "model_{}.pth".format(self.identifier)))


if __name__ == '__main__':
    conf_file = 'conf_file.yml'
    setup = Setup(conf_file)
    #logging.basicConfig(level=logging.DEBUG)
    logging.debug("Running with configuration file")

    setup.run()
    for c in setup.server.list_of_clients:
        get_accuracy(c.model, c.test_loader)
        plot_losses(c.train_loss, c.test_loss)
      #  c.architecture.get_confusion_matrix()
    print('ok')
    #if setup.to_save:
     #   setup.save()