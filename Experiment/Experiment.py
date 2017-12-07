import torch
from torch.autograd import Variable
from pathlib import Path
import os
from tqdm import tqdm


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


    def run(self):
        self.trainer_delegate.on_experiment_start()
        for data_split in self.data_source_delegate.retrieve_dataset():
            for epoch in range(self.n_epochs):
                train, val = self.trainer_delegate.on_epoch_start(data_split)
                for data in tqdm(train):
                    self.model.train()
                    model_input = self.trainer_delegate.create_model_input(data)
                    model_output = self.trainer_delegate.create_model_output(model_input, self.model)
                    transformed_output = self.trainer_delegate.apply_output_transformation(model_output,
                                                                                           self.output_transformation)
                    labels = self.trainer_delegate.create_data_labels(data)
                    model_loss = self.trainer_delegate.calculate_loss(self.loss_function, transformed_output,
                                                                      labels, mode="TRAIN")
                    self.trainer_delegate.apply_backward_pass(self.optimizer, model_loss, self.model)

                self.trainer_delegate.on_finish_train(self.model, epoch)

                for data in val:
                    self.model.eval()
                    model_input = self.trainer_delegate.create_model_input(data)
                    model_output = self.trainer_delegate.create_model_output(model_input, self.model)
                    transformed_output = self.trainer_delegate.apply_output_transformation(model_output,
                                                                                           self.output_transformation)
                    labels = self.trainer_delegate.create_data_labels(data)
                    _ = self.trainer_delegate.calculate_loss(self.loss_function, transformed_output,
                                                                      labels, mode="EVAL")

                self.trainer_delegate.on_finish_validation(self.model, epoch)
                self.trainer_delegate.on_epoch_end()
            self.trainer_delegate.on_end_data_split()
        self.trainer_delegate.on_end_experiment()

