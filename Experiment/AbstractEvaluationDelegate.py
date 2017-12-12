from abc import ABC, abstractmethod

class AbstractEvaluationDelegate(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def on_setup_model(self, model):
        pass

    @abstractmethod
    def on_test_start(self, dataset):
        pass

    @abstractmethod
    def on_test_end(self):
        pass
