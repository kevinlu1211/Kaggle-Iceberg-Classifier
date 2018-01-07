from src.utils.cuda import cudarize
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from pathlib import Path

class TrainerDelegate(object):

    def __init__(self, experiment_id, study_save_path):
        self.experiment_id = experiment_id
        self.study_save_path = study_save_path

    def on_epoch_start(self, dataset):
        train = dataset['train']
        val = dataset['val']
        return train, val

    def create_model_input(self, data):
        return cudarize(Variable(data['input']))

    def create_data_labels(self, data):
        return cudarize(Variable(data['label']))

    def create_model_output(self, model_input, model):
        model_output = model(model_input)
        return model_output

    def apply_output_transformation(self, model_output):
        output = F.sigmoid(model_output)
        return output

    def calculate_loss(self, loss_function, model_output, labels):
        loss = loss_function(model_output, labels)
        return loss

    def apply_backward_pass(self, optimizer, model_loss):
        optimizer.zero_grad()
        model_loss.backward()
        optimizer.step()

    def update_scheduler(self, validation_loss, scheduler):
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation_loss.data[0])
            else:
                scheduler.step()
