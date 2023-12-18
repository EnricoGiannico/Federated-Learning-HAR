import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, \
  TensorDataset
from torch.nn.utils import rnn as rnn_utils
import pandas as pd
from datetime import datetime
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt


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

    def set_state(self, checkpoint):
        lstm_state = checkpoint['lstm']
        fc_state = checkpoint['fc']
        self.lstm.load_state_dict(lstm_state)
        self.fc.load_state_dict(fc_state)


    def save_state(self, filename):
        state = self.get_state()
        torch.save(state, filename)

    def load_state(self, filename):
        checkpoint = torch.load(filename)
        self.lstm.load_state_dict(checkpoint['lstm'])
        self.fc.load_state_dict(checkpoint['fc'])



for client in range(1, 37):
    data = torch.load(f'..\data\lstm\dataset_lstm_{client}.pt')
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

    input_size = 3  # Number of features
    hidden_size = 50  # Number of features in hidden state
    num_classes = 6  # Number of output classes

    model = LSTMClassifier(input_size, hidden_size, num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    loss_values = []
    num_epoch = 15

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
        loss_values.append(average_loss)
        print(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {average_loss:.4f}')

    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Training Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Test the model
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
    print(f'Accuracy client_{client}: {accuracy:.4f}')