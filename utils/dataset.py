from torchvision import transforms
import pandas as pd
import torch
import numpy as np
import logging
from copy import deepcopy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
use_cuda = torch.cuda.is_available()


def load_data(data_fp):
    data = pd.read_json(data_fp)
    return shuffle(data)


def create_train_val_dataloaders(data_fp, train_size=0.8,  batch_size=128):
    data = load_data(data_fp)
    train_df, val_df = train_test_split(data, train_size=train_size)

    # Deep copy the views of the df so that changes in the original df won't affect the current df
    # but more importantly it gets rid of CopyWarning error
    train_df = deepcopy(train_df)
    val_df = deepcopy(val_df)
    train_loader = create_dataloader(train_df, is_train=True, batch_size=batch_size)
    val_loader = create_dataloader(val_df, is_train=False, batch_size=batch_size)
    return train_loader, val_loader


def create_dataloader_from_path(data_fp, is_train=True, batch_size=64):
    df = pd.read_json(data_fp)
    dataset = IcebergDataset(df, is_train=is_train)
    return DataLoader(dataset, batch_size=batch_size)

def create_dataloader(df, is_train=True, batch_size=64):
    dataset = IcebergDataset(df, is_train=is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class IcebergDataset(torch.utils.data.Dataset):
    def __init__(self, df, is_train=True):
        super().__init__()
        img, target, ids = self.preprocess_data(df)
        self.img = img
        self.target = target
        self.ids = ids
        self.is_train = is_train
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomCrop(60),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomCrop(60),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

    def preprocess_data(self, data):
        logging.info("Preprocessing data ...")
        logging.info("Reshaping input images ...")
        data['band_1_rs'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
        data['band_2_rs'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
        data['band_3_rs'] = (data['band_1_rs'] + data['band_2_rs'])/2
        data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')

        band_1 = np.concatenate([im for im in data['band_1_rs']]).reshape(-1, 75, 75)
        band_2 = np.concatenate([im for im in data['band_2_rs']]).reshape(-1, 75, 75)
        band_3 = np.concatenate([im for im in data['band_3_rs']]).reshape(-1, 75, 75)

        logging.info("Converting training data to Tensors ...")

        # Batch, Height, Width, Channel
        img = np.stack([band_1, band_2, band_3], axis=3)
        img_max = np.max(img, keepdims=True, axis=(1, 2))
        img_min = np.min(img, keepdims=True, axis=(1, 2))
        max_min_diff = img_max - img_min
        img_uint8 = (((img - img_min)/max_min_diff) * 255).astype(np.uint8)

        if 'is_iceberg' in data:
            target = self.convert_target_to_tensor(data['is_iceberg'].values)
        else:
            target = self.convert_target_to_tensor([-1] * data.shape[0])
        id = data['id'].tolist()
        return img_uint8, target, id

    def convert_target_to_tensor(self, target):
        target = np.expand_dims(target, 1)  # Must be reshaped for BCE loss
        return torch.from_numpy(target).type(torch.FloatTensor) # Must be float for BCE loss

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        return (self.transform(self.img[i]), self.target[i]), self.ids[i]
