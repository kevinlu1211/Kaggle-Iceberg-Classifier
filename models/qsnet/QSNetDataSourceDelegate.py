from src.Experiment import DataSourceDelegate

class QSNetDataSourceDelegate(DataSourceDelegate):

    def __init__(self, training_data_path, testing_data_path, batch_size):
        super().__init__(training_data_path, testing_data_path, batch_size)


