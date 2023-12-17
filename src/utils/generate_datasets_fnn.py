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

y_encoding = {
    "Downstairs": 0,
    "Jogging": 1,
    "Sitting": 2,
    "Standing": 3,
    "Upstairs": 4,
    "Walking": 5
}

def load_data_with_timestamp():
  data = np.loadtxt("../../data/dataset.txt", delimiter=',', dtype=object)
  data = data[:, [0, 1, 2, 3, 4, 5]]
  data_pd = pd.DataFrame(data, columns=['client', 'activity', 'timestamp', 'x_acc', 'y_acc', 'z_acc'])
  data_pd['x_acc'] = data_pd['x_acc'].astype(float)
  data_pd['y_acc'] = data_pd['y_acc'].astype(float)
  data_pd['z_acc'] = data_pd['z_acc'].astype(float)
  data_pd['client'] = data_pd['client'].astype(int)
  return data_pd


def convert_timestamp_to_string(timestamp):
  timestamp = int(timestamp) // 1_000_000
  datetime_object = datetime.utcfromtimestamp(timestamp)
  return datetime_object.strftime('%Y-%m-%d %H:%M:%S')

DATA_TIMESTAMP = load_data_with_timestamp()
df = DATA_TIMESTAMP.drop_duplicates()

sequence_length = 3


x_tensor_list = []
y_tensor_list = []
for client in df['client'].unique().tolist():
    for activity in df['activity'].loc[df['client'] == client].unique().tolist():
        dg = df.loc[(df['client'] == client) & (df['activity'] == activity)]
        dg['date_time'] = dg['timestamp'].apply(lambda x: convert_timestamp_to_string(x))
        dg['date_time'] = pd.to_datetime(dg['date_time'])
        dg['time_diff'] = dg['date_time'].shift(-1) - dg['date_time']
        dg['time_diff'].fillna(pd.Timedelta(seconds=0), inplace=True)
        dg = dg.loc[dg['time_diff'] <= pd.Timedelta(seconds=120)]
        sequences = []
        for i in range(0, len(dg) - sequence_length + 1, 3):
            sequence = dg[['x_acc', 'y_acc', 'z_acc']].iloc[i:i + sequence_length].values
            sequences.append(sequence)
        y = np.full(len(sequences), y_encoding[activity])
        y_tensor = torch.Tensor(y).long()  # Use .long() for integer labels
        fnn_input = np.array(sequences)
        x_tensor = torch.Tensor(fnn_input)
        x_tensor_list.append(x_tensor)
        y_tensor_list.append(y_tensor)

x_combined = torch.cat(x_tensor_list, dim=0)
y_combined = torch.cat(y_tensor_list, dim=0)
x_combined = x_combined.view(-1, 3 * 3)
combined_dataset = TensorDataset(x_combined, y_combined)
torch.save(combined_dataset, '..\..\data\\fnn\dataset_fnn.pt')

print('ok')
for client in df['client'].unique().tolist():
    x_tensor_list = []
    y_tensor_list = []
    for activity in df['activity'].loc[df['client'] == client].unique().tolist():
        dg = df.loc[(df['client'] == client) & (df['activity'] == activity)]
        dg['date_time'] = dg['timestamp'].apply(lambda x: convert_timestamp_to_string(x))
        dg['date_time'] = pd.to_datetime(dg['date_time'])
        dg['time_diff'] = dg['date_time'].shift(-1) - dg['date_time']
        dg['time_diff'].fillna(pd.Timedelta(seconds=0), inplace=True)
        dg = dg.loc[dg['time_diff'] <= pd.Timedelta(seconds=120)]
        sequences = []
        for i in range(0, len(dg) - sequence_length + 1, 3):
            sequence = dg[['x_acc', 'y_acc', 'z_acc']].iloc[i:i + sequence_length].values
            sequences.append(sequence)
        y = np.full(len(sequences), y_encoding[activity])
        y_tensor = torch.Tensor(y).long()
        fnn_input = np.array(sequences)
        x_tensor = torch.Tensor(fnn_input)
        x_tensor_list.append(x_tensor)
        y_tensor_list.append(y_tensor)
    x_combined = torch.cat(x_tensor_list, dim=0)
    y_combined = torch.cat(y_tensor_list, dim=0)
    x_combined = x_combined.view(-1, 3 * 3)
    combined_dataset = TensorDataset(x_combined, y_combined)
    torch.save(combined_dataset, f'..\..\data\\fnn\dataset_fnn_{client}.pt')

