from abc import ABC, abstractmethod

class AbstractDataSourceDelegate(ABC):
    def __init__(self):
        pass


    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess_data(self, data):
        pass

    @abstractmethod
    def data_split(self, data):
        pass

    @abstractmethod
    def retrieve_dataset_for_train(self):
        pass

    @abstractmethod
    def retrieve_dataset_for_test(self):
        pass












