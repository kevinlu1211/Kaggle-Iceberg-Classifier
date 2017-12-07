import torch
import os
from pathlib import Path
import numpy as np
from src.experiment import TrainerDelegate
import logging


class QSNetTrainerDelegate(TrainerDelegate):

    def __init__(self, id, experiment_save_path):
        super().__init__(id, experiment_save_path)








