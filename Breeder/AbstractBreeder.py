from abc import ABC, abstractmethod


class AbstractBreeder(ABC):
    def __init__(self, config, experiment_factory):
        self.config = config
        self.experiment_factory = experiment_factory

    @abstractmethod
    def create_experiment_config(self):
        pass

    @abstractmethod
    def start_breeding(self):
        pass
