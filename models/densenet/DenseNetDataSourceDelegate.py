from src.Experiment import DataSourceDelegate


class DenseNetDataSourceDelegate(DataSourceDelegate):

    def __init__(self, training_data_path, testing_data_path, batch_size, splits_to_use, n_splits):
        super().__init__(training_data_path, testing_data_path, batch_size, splits_to_use, n_splits)


