from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict

class AbstractSaverDelegate(ABC):

    def __init__(self):
        self.training_results = defaultdict(OrderedDict)
        self.validation_results = defaultdict(OrderedDict)
        self.training_loss_for_epoch = []
        self.validation_loss_for_epoch = []

    @abstractmethod
    def on_model_loss(self, loss):
        pass

    @abstractmethod
    def on_epoch_end(self, model, loss):
        pass



