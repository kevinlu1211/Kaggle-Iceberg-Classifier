from abc import ABC, abstractmethod

class AbstractDataSourceDelegate(ABC):
    def __init__(self, training_data_path):
        self.training_data_path = training_data_path
        self.data = None


    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abstractmethod
    def preprocess_data(self, data):
        raise NotImplementedError

    @abstractmethod
    def data_split(self, data):
        raise NotImplementedError

    @abstractmethod
    def retrieve_dataset_for_train(self):
        raise NotImplementedError

    @abstractmethod
    def retrieve_dataset_for_test(self):
        raise NotImplementedError












