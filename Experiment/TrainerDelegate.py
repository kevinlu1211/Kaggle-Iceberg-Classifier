from src.utils.cuda import cudarize
import torch
from torch.autograd import Variable
import torch.functional as F
from pathlib import Path

class TrainerDelegate(object):

    def __init__(self, experiment_id, study_save_path):
        self.experiment_id = experiment_id
        self.study_save_path = study_save_path

    def on_experiment_start(self, model):
        # This is done so that the initial weights of the model never change after each CV split
        pth = Path("tmp")
        pth.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f"tmp/{self.experiment_id}.pth")

    def on_setup_model(self, model):
        model.load_state_dict(torch.load(f"tmp/{self.experiment_id}.pth"))
        return model

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


    def on_model_output(self, model_output):
        pass

    def apply_backward_pass(self, optimizer, model_loss):
        optimizer.zero_grad()
        model_loss.backward()
        optimizer.step()

    def on_epoch_end(self, model, epoch, fold_num):
        pass

    def on_end_experiment(self):
        pass


