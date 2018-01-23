from src.utils.cuda import cudarize
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from pathlib import Path

## TODO: Probably cleaner to superclass/subclass, will refactor in next Kaggle competition

class TrainerDelegate(object):

    def __init__(self, experiment_id, study_save_path, model_output_method="Standard"):
        self.experiment_id = experiment_id
        self.study_save_path = study_save_path
        self.model_output_handler = model_output_handlers[model_output_method]

    def on_epoch_start(self, dataset):
        train = dataset['train']
        val = dataset['val']
        return train, val

    def create_data_labels(self, data):
        return cudarize(Variable(data['label']))

    def create_model_output(self, data, model):
        model_output = self.model_output_handler(data, model)
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

def create_model_output_with_image_statistics(data, model):
    model_inp_image = cudarize(Variable(data['image']))
    model_inp_image_stats = cudarize(Variable(data['image_stats']))
    model_output = model(model_inp_image, model_inp_image_stats)
    return model_output

def create_model_output_standard(data, model):
    model_input = cudarize(Variable(data['input']))
    model_output = model(model_input)
    return model_output

model_output_handlers = {
    "Standard": create_model_output_standard,
    "ImageStatistics": create_model_output_with_image_statistics
}
