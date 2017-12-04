import torch
from torch.autograd import Variable
from pathlib import Path
import os


class Experiment(object):
    def __init__(self, n_epochs, data_source_delegate, trainer_delegate,
                 model, output_transformation, loss_function, optimizer):
        self.n_epochs = n_epochs
        self.data_source_delegate = data_source_delegate
        self.trainer_delegate = trainer_delegate
        self.model = model
        self.output_transformation = output_transformation
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.experiment_outputs = []
        self.experiment_losses = []


    @property
    def name(self):
        return f"{self.config['model_name']}-{self.config['id']}"

    def run(self):
        for dataset in self.data_source_delegate.retrieve_dataset():
            for epoch in range(self.n_epochs):
                train, val = self.trainer_delegate.on_epoch_start(dataset)
                for data in train:
                    model_input = self.trainer_delegate.create_model_input(data)
                    model_output = self.trainer_delegate.create_model_output(model_input, self.model)
                    transformed_output = self.trainer_delegate.apply_output_transformation(model_output,
                                                                                           self.output_transformation)
                    labels = self.trainer_delegate.create_data_labels(data)
                    model_loss = self.trainer_delegate.calculate_loss(self.loss_function, transformed_output, labels, mode="TRAIN")
                    self.trainer_delegate.apply_backward_pass(self.optimizer, model_loss, self.model)

                self.trainer_delegate.on_finish_train(self.model, epoch)

                for data in val:
                    model_input = self.trainer_delegate.create_model_input(data)
                    model_output = self.trainer_delegate.create_model_output(model_input, self.model)
                    transformed_output = self.trainer_delegate.apply_output_transformation(model_output,
                                                                                           self.output_transformation)
                    labels = self.trainer_delegate.create_data_labels(data)
                    model_loss = self.trainer_delegate.calculate_loss(self.loss_function, transformed_output, labels, mode="EVAL")

                self.trainer_delegate.on_finish_validation(self.model, epoch)

