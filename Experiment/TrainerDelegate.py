from src.utils.cuda import cudarize
import torch
from torch.autograd import Variable
import os
from pathlib import Path
import numpy as np
import logging
from .AbstractTrainerDelegate import AbstractTrainerDelegate
from copy import deepcopy


class TrainerDelegate(AbstractTrainerDelegate):

    def __init__(self, experiment_id, experiment_path):
        self.experiment_id = experiment_id
        self.experiment_path = experiment_path
        self.total_training_loss = []
        self.total_validation_loss = []
        self.training_loss_for_epoch = []
        self.validation_loss_for_epoch = []

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
        self.training_loss_for_epoch = []
        self.validation_loss_for_epoch = []
        return train, val

    def create_model_input(self, data):
        return cudarize(Variable(data['input']))

    def create_data_labels(self, data):
        return cudarize(Variable(data['label']))

    def create_model_output(self, model_input, model):
        model_output = model(model_input)
        return model_output

    def apply_output_transformation(self, model_output, output_transformer):
        if output_transformer is not None:
            output = output_transformer(model_output)
        else:
            output = model_output
        return output

    def on_model_output(self, model_output):
        pass

    def calculate_loss(self, loss_function, model_output, labels, mode):
        loss = loss_function(model_output, labels)
        if mode == "TRAIN":
            self.training_loss_for_epoch.append(loss.data)
        else:
            self.validation_loss_for_epoch.append(loss.data)
        return loss

    def apply_backward_pass(self, optimizer, model_loss, model):
        optimizer.zero_grad()
        model_loss.backward()
        optimizer.step()

    def on_epoch_end(self, model, epoch):
        avg_loss, = np.mean(np.array(self.training_loss_for_epoch)).cpu().numpy()
        logging.info(f"Training loss for epoch: {epoch} is {avg_loss}")
        avg_loss, = np.mean(np.array(self.validation_loss_for_epoch)).cpu().numpy()
        logging.info(f"Validation loss for epoch: {epoch} is {avg_loss}")

        models_checkpoint_folder_path = Path(f"{self.experiment_path}/{self.experiment_id}/model_checkpoints")
        models_checkpoint_folder_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), os.fspath(f"{models_checkpoint_folder_path}/{avg_loss}.pth"))


    def on_finish_data_split(self):
        self.total_training_loss.append(self.training_loss_for_epoch)
        self.total_validation_loss.append(self.validation_loss_for_epoch)

    def on_end_experiment(self):
        pass




