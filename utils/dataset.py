import pandas as pd
import torch
import numpy as np
import logging
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader


def load_data(data_fp):
    data = pd.read_json(data_fp)
    return shuffle(data)

def preprocess_data(data):
    data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
    data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
    data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')
    band_1 = np.concatenate([im for im in data['band_1']]).reshape(-1, 75, 75)
    band_2 = np.concatenate([im for im in data['band_2']]).reshape(-1, 75, 75)

    # Batch, Channel, Height, Width
    full_img = np.stack([band_1, band_2], axis=1)
    train =convert_train_to_tensor(full_img)
    target = convert_target_to_tensor(data['is_iceberg'].values)
    return train, target


def convert_train_to_tensor(train):
    train = np.array(train, dtype=np.float32)
    return torch.from_numpy(train)


def convert_target_to_tensor(target):
    target = np.expand_dims(target, 1)  # Must be reshaped for PyTorch!
    return torch.from_numpy(target).type(torch.FloatTensor)


def create_dataloader(data_fp, use_cuda, batch_size=128):
    data = load_data(data_fp)
    train_data, target_data = preprocess_data(data)
    if use_cuda:
        train_data = train_data.cuda()
        target_data = target_data.cuda()

    dataset = TensorDataset(train_data, target_data)
    train, val = train_test_split(dataset)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader





class FullTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, full_ds, offset, length):
        self.full_ds = full_ds
        self.offset = offset
        self.length = length
        assert len(full_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(FullTrainingDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.full_ds[i + self.offset]

def train_test_split(dataset, val_share=0.11):
    val_offset = int(len(dataset)*(1-val_share))
    print("Offset:" + str(val_offset))
    return FullTrainingDataset(dataset, 0, val_offset),\
        FullTrainingDataset(dataset, val_offset, len(dataset)-val_offset)
