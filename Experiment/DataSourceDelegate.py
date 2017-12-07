import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import create_dataloader
from .AbstractDataSourceDelegate import AbstractDataSourceDelegate


class DataSourceDelegate(AbstractDataSourceDelegate):
    def __init__(self, training_data_path, batch_size):
        self.training_data_path = training_data_path
        self.batch_size = batch_size
        self.data = None
        self.setup()

    def _get_data_reader(self, path):
        file_extension = path.split(".")[-1]
        return {
            "csv": pd.read_csv,
            "json": pd.read_json
        }.get(file_extension)

    def setup(self):
        data = self.load_data()
        preprocessed_data = self.preprocess_data(data)
        self.data = self.data_split(preprocessed_data)

    def load_data(self):
        path = self.training_data_path
        assert path is not None
        data_reader = self._get_data_reader(path)
        return data_reader(path)

    def preprocess_data(self, data):
        return data

    def data_split(self, data):
        """
        By default this uses a train/test split of 80/20
        :param data: This whole dataset
        :return: a list of iterables which will probably contain splits of the dataset, in the most simple case, it
        would be a train/test split, but could also be splits from a KFold split
        """
        train_df, val_df = train_test_split(data, test_size=0.2)
        split_data = [(train_df, val_df)]
        return split_data

    def retrieve_dataset(self):
        for d in self.data:
            train_df, val_df = d
            train_dataloader = create_dataloader(train_df, is_train=True,
                                                 batch_size=self.batch_size)
            val_dataloader = create_dataloader(val_df, is_train=False,
                                               shuffle=False,
                                               batch_size=self.batch_size)
            yield {"train": train_dataloader,
                   "val": val_dataloader}












