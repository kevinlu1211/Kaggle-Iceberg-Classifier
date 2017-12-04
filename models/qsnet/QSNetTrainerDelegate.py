from src.experiment import TrainerDelegate
from torch.autograd import Variable
import torch

class QSNetTrainerDelegate(TrainerDelegate):

    def __init__(self, experiment_save_path):
        super().__init__(experiment_save_path)
        self.use_cuda = torch.cuda.is_available()

    def create_model_input(self, data):
        model_input = Variable(data["input"])
        if self.use_cuda:
            return model_input.cuda()
        else:
            return model_input

    def create_data_labels(self, data):
        label = Variable(data["label"])
        if self.use_cuda:
            return label.cuda()
        else:
            return label

    def on_epoch_start(self, dataset):
        train, val = dataset
        return train, val







