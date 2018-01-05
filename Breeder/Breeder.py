from abc import ABC, abstractmethod
from .AbstractBreeder import AbstractBreeder

class Breeder(AbstractBreeder):
    def __init__(self, config, experiment_factory, ):
        self.config = config
        self.experiment_factory = experiment_factory

