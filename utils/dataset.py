from torchvision import transforms
import pandas as pd
import torch
import numpy as np
import logging
from copy import deepcopy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import sklearn

use_cuda = torch.cuda.is_available()


def load_data(data_fp):
    data = pd.read_json(data_fp)
    return shuffle(data)


def create_train_val_dataloaders(data_fp, train_size=0.8,  batch_size=128):
    data = load_data(data_fp)
    train_df, val_df = train_test_split(data, train_size=train_size)

    train_df = deepcopy(train_df)
    val_df = deepcopy(val_df)
    train_loader = create_dataloader(train_df, is_train=True, batch_size=batch_size)
    val_loader = create_dataloader(val_df, is_train=False, batch_size=batch_size)
    return train_loader, val_loader


def create_dataloader_from_path(data_fp, is_train=True, batch_size=64):
    df = pd.read_json(data_fp)
    dataset = IcebergDataset(df, is_train=is_train)
    return DataLoader(dataset, batch_size=batch_size)

def create_dataloader(df, image_size, is_train=True, shuffle=True, batch_size=64):
    dataset = IcebergDataset(df, is_train, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class IcebergDataset(torch.utils.data.Dataset):
    def __init__(self, df, is_train, image_size):
        super().__init__()
        self.img = df['input']
        self.target = df['label']
        self.ids = df['id']
        self.is_train = is_train

        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                # transforms.RandomCrop(60),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomCrop(60),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

    def convert_target_to_tensor(self, target):
        target = np.expand_dims(target, 1)  # Must be reshaped for BCE loss
        return torch.from_numpy(target).type(torch.FloatTensor)  # Must be float for BCE loss

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        return {"input": self.transform(self.img[i]),
                "label": self.convert_target_to_tensor(self.target[i]),
                "id": self.ids[i]}
