from src.utils.cuda import cudarize
from collections import defaultdict, OrderedDict
import torch
from torch.autograd import Variable
import os
from pathlib import Path
import numpy as np
import logging
from .AbstractTrainerDelegate import AbstractTrainerDelegate
import json
import torch.optim.lr_scheduler

class TrainerDelegate(AbstractTrainerDelegate):

    def __init__(self, experiment_id, study_save_path):
        self.experiment_id = experiment_id
        self.study_save_path = study_save_path
        self.validation_results = defaultdict(OrderedDict)
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

    def update_scheduler(self, validation_loss, scheduler):
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation_loss.data[0])
            else:
                scheduler.step()
        # for param_group in optimizer.param_groups:
        #     lr = param_group['lr']
        # print(f"lr: {lr}")

    def apply_backward_pass(self, optimizer, scheduler, model_loss, model):
        optimizer.zero_grad()
        model_loss.backward()
        optimizer.step()


    def on_epoch_end(self, model, epoch, fold_num):
        pass

    def on_end_experiment(self):
        pass


