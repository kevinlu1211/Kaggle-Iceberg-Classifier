import random
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import shuffle
from torchvision import transforms
import pandas as pd
import torch
import numpy as np
import logging
from copy import deepcopy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import imutils

from torch.utils.data import TensorDataset, DataLoader
import functools


def compose(*functions):
    def compose2(f, g):
        return lambda x: f(g(x))

    return functools.reduce(compose2, functions, lambda x: x)


class DataSourceDelegate(object):
    def __init__(self, training_data_path, testing_data_path, batch_size,
                 splits_to_use, n_splits, data_handler_method="NormalizeThreeChannels",
                 image_size=(75, 75)):
        self.training_data_path = training_data_path
        self.test_data_path = testing_data_path
        self.batch_size = batch_size
        self.splits_to_use = splits_to_use
        self.n_splits = n_splits
        self.splits = None
        self.training_data = None
        self.test_data = None
        self.image_size = image_size
        self.data_handler = data_handlers[data_handler_method]

    def _get_data_reader(self, path):
        file_extension = path.split(".")[-1]
        return {
            "csv": pd.read_csv,
            "json": pd.read_json
        }.get(file_extension)

    def setup_training_data(self):
        assert self.training_data_path is not None
        training_data = self.load_data(self.training_data_path)
        preprocessed_training_data = self.data_handler.preprocess_data(training_data)
        shuffle(preprocessed_training_data)
        self.splits = self.data_split(training_data)
        self.training_data = preprocessed_training_data

    def setup_test_data(self):
        assert self.training_data_path is not None
        test_data = self.load_data(self.test_data_path)
        preprocessed_test_data = self.data_handler.preprocess_data(test_data)
        self.test_data = preprocessed_test_data

    def load_data(self, path):
        assert path is not None
        data_reader = self._get_data_reader(path)
        return data_reader(path)

    def data_split(self, data):
        folds = StratifiedKFold(n_splits=self.n_splits).split(data, data['is_iceberg'])
        return random.sample(list(folds), self.splits_to_use)

    def retrieve_dataset_for_train(self):
        if self.training_data is None:
            self.setup_training_data()

        for train_idx, test_idx in self.splits:
            train_df = self.training_data.iloc[train_idx].reset_index(drop=True)
            val_df = self.training_data.iloc[test_idx].reset_index(drop=True)
            train_dataloader = self.data_handler.create_dataloader(train_df, is_train=True,
                                                                   shuffle=True,
                                                                   batch_size=self.batch_size,
                                                                   image_size=self.image_size)
            val_dataloader = self.data_handler.create_dataloader(val_df, is_train=False,
                                                                 shuffle=False,
                                                                 batch_size=self.batch_size,
                                                                 image_size=self.image_size)
            yield {"train": train_dataloader,
                   "val": val_dataloader}

    def retrieve_dataset_for_test(self):
        if self.test_data is None:
            self.setup_test_data()
        test_df = self.test_data
        test_dataloader = self.data_handler.create_dataloader(test_df, is_train=False,
                                                              shuffle=False,
                                                              batch_size=self.batch_size,
                                                              image_size=self.image_size)
        yield {"test": test_dataloader}


class NormalizeThreeChannels(object):
    def __init__(self):
        pass
    @staticmethod
    def preprocess_data(data):

        logging.info("Preprocessing data ...")
        logging.info("Reshaping input images ...")
        data['band_1_rs'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
        data['band_2_rs'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
        data['band_3_rs'] = (data['band_1_rs'] + data['band_2_rs']) / 2
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
        imgs_uint8 = (((img - img_min) / max_min_diff) * 255).astype(np.uint8)

        if 'is_iceberg' in data:
            targets = data['is_iceberg'].values
        else:
            targets = [-1] * data.shape[0]

        img_ids = data['id'].tolist()
        df_dict = []
        for img, target, img_id in zip(imgs_uint8, targets, img_ids):
            df_dict.append({
                "input": img,
                "label": target,
                "id": img_id
            })

        df = pd.DataFrame(df_dict)
        return df

    def get_transforms(self, image_size):
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        return train_transform, test_transform

    def create_dataloader(self, df, image_size, is_train=True, shuffle=True, batch_size=64):
        train_transform, test_transform = self.get_transforms(image_size)
        dataset = IcebergDataset(df, train_transform, test_transform, is_train, image_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class ThreeChannels(object):
    def __init__(self):
        pass

    @staticmethod
    def preprocess_data(data):

        logging.info("Preprocessing data ...")
        logging.info("Reshaping input images ...")
        data['band_1_rs'] = data['band_1'].apply(lambda x: np.array(x).reshape(-1, 75, 75))
        data['band_2_rs'] = data['band_2'].apply(lambda x: np.array(x).reshape(-1, 75, 75))
        data['band_3_rs'] = (data['band_1_rs'] + data['band_2_rs']) / 2
        data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')
        data['inc_angle'].fillna(0, inplace=True)

        band_1 = np.concatenate([im for im in data['band_1_rs']]).reshape(-1, 75, 75)
        band_2 = np.concatenate([im for im in data['band_2_rs']]).reshape(-1, 75, 75)
        band_3 = np.concatenate([im for im in data['band_3_rs']]).reshape(-1, 75, 75)
        logging.info("Converting training data to Tensors ...")

        # Batch, Height, Width, Channel
        imgs = np.stack([band_1, band_2, band_3], axis=1).astype(np.float32)
        if 'is_iceberg' in data:
            targets = data['is_iceberg'].values
        else:
            targets = [-1] * data.shape[0]

        img_ids = data['id'].tolist()
        df_dict = []
        for img, target, img_id in zip(imgs, targets, img_ids):
            df_dict.append({
                "input": img,
                "label": target,
                "id": img_id
            })

        df = pd.DataFrame(df_dict)
        return df

    def get_transforms(self, image_size):
        train_transform = transforms.Compose([
            horizontal_flip,
            rotation,
            convert_image_to_tensor
        ])

        test_transform = transforms.Compose([
            convert_image_to_tensor
        ])
        return train_transform, test_transform

    def create_dataloader(self, df, image_size, is_train=True, shuffle=True, batch_size=64):
        train_transform, test_transform = self.get_transforms(image_size)
        dataset = IcebergDataset(df, train_transform, test_transform, is_train)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class TwoChannels(object):
    def __init__(self):
        pass

    @staticmethod
    def preprocess_data(data):

        logging.info("Preprocessing data ...")
        logging.info("Reshaping input images ...")
        data['band_1_rs'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
        data['band_2_rs'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
        data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')
        data['inc_angle'].fillna(0, inplace=True)

        band_1 = np.concatenate([im for im in data['band_1_rs']]).reshape(-1, 75, 75)
        band_2 = np.concatenate([im for im in data['band_2_rs']]).reshape(-1, 75, 75)

        logging.info("Converting training data to Tensors ...")

        # Batch, Height, Width, Channel
        imgs = np.stack([band_1, band_2], axis=1)
        if 'is_iceberg' in data:
            targets = data['is_iceberg'].values
        else:
            targets = [-1] * data.shape[0]

        img_ids = data['id'].tolist()
        df_dict = []
        for img, target, img_id in zip(imgs, targets, img_ids):
            df_dict.append({
                "input": img,
                "label": target,
                "id": img_id
            })

        df = pd.DataFrame(df_dict)
        return df

    def get_transforms(self, image_size):
        train_transform = transforms.Compose([
            horizontal_flip,
            vertical_flip,
            convert_image_to_tensor
        ])

        test_transform = transforms.Compose([
            convert_image_to_tensor
        ])
        return train_transform, test_transform


    def create_dataloader(self, df, image_size, is_train=True, shuffle=True, batch_size=64):
        train_transform, test_transform = self.get_transforms(image_size)
        dataset = IcebergDataset(df, train_transform, test_transform, is_train)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class IcebergDataset(torch.utils.data.Dataset):
    def __init__(self, df, is_train, train_transform, test_transform):
        super().__init__()
        self.img = df['input']
        self.target = df['label']
        self.ids = df['id']
        self.is_train = is_train
        if is_train:
            self.transform = train_transform
        else:
            self.transform = test_transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        return {"input": self.transform(self.img[i]),
                "label": convert_target_to_tensor(self.target[i]),
                "id": self.ids[i]}


def convert_target_to_tensor(target):
    target = np.expand_dims(target, 1)  # Must be reshaped for BCE loss
    return torch.from_numpy(target).type(torch.FloatTensor)  # Must be float for BCE loss


def convert_image_to_tensor(image):
    return torch.from_numpy(image).type(torch.FloatTensor)

def horizontal_flip(image):
    if random.random() > 0.5:
        return cv2.flip(image.copy(), 0)
    return image

def vertical_flip(image):
    if random.random() > 0.5:
        return cv2.flip(image.copy(), 1)
    return image


def rotation(image, degrees=45):
    if random.random() > 0.5:
        orientation = random.sample([-1, 1], 1)[0]
        degrees = random.random() * degrees
        return imutils.rotate_bound(image, orientation * degrees)
    return image

data_handlers = {
    "NormalizeThreeChannels": NormalizeThreeChannels(),
    "TwoChannels": TwoChannels(),
    "ThreeChannels": ThreeChannels()
}
