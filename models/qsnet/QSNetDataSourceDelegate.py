from src.experiment import DataSourceDelegate
from src.utils import create_dataloader

class QSNetDataSourceDelegate(DataSourceDelegate):

    def __init__(self, training_data_path, batch_size):
        super().__init__(training_data_path, batch_size)


