from experiment import DataSourceDelegate


class DenseNetDataSourceDelegate(DataSourceDelegate):

    def __init__(self, config, use_cuda):
        super().__init__(config, use_cuda)
