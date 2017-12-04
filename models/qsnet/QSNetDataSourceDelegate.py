from src.experiment import DataSourceDelegate
from src.utils import create_dataloader

class QSNetDataSourceDelegate(DataSourceDelegate):

    def __init__(self, training_data_path, batch_size):
        super().__init__(training_data_path)
        self.batch_size = batch_size

    def retrieve_dataset(self):
        for d in self.data:
            train_df, val_df = d
            train_dataloader = create_dataloader(train_df, is_train=True,
                                                 batch_size=self.batch_size)
            val_dataloader = create_dataloader(val_df, is_train=False,
                                               shuffle=False,
                                               batch_size=self.batch_size)
            yield (train_dataloader, val_dataloader)
