from src.utils.cuda import cudarize
import torch
from torch.autograd import Variable
import os
from pathlib import Path
import numpy as np
import logging

class TrainerDelegate(object):

    def __init__(self, id, experiment_save_path):
        self.id = id
        self.experiment_save_path = experiment_save_path
        self.total_training_loss = []
        self.total_validation_loss = []
        self.training_loss_for_epoch = []
        self.validation_loss_for_epoch = []

    def on_experiment_start(self):
        pass

    def on_new_data_split(self, data_split):
        pass

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
            return output_transformer(model_output)
        else:
            return model_output

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

    def on_finish_train(self, model, epoch):
        avg_loss, = np.mean(np.array(self.training_loss_for_epoch)).numpy()
        logging.info(f"Training loss for epoch: {epoch} is {avg_loss}")

    def on_finish_validation(self, model, epoch):
        avg_loss, = np.mean(np.array(self.validation_loss_for_epoch)).numpy()
        logging.info(f"Validation loss for epoch: {epoch} is {avg_loss}")

        models_checkpoint_folder_path = Path(f"{self.experiment_save_path}/saved_models/{self.id}")
        models_checkpoint_folder_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), os.fspath(f"{models_checkpoint_folder_path}/{avg_loss}.pth"))

    def on_epoch_end(self):
        self.total_training_loss.append(self.training_loss_for_epoch)
        self.total_validation_loss.append(self.validation_loss_for_epoch)

    def on_end_data_split(self):
        pass

    def on_end_experiment(self):
        pass

    # def on_load_model(self, experiment_path, ckpt_name, for_eval):
    #     assert ckpt_name is not None
    #     model_id = self.metadata["id"]
    #     model_checkpoint_path = f"{experiment_path}/saved_models/{model_id}/{ckpt_name}"
    #     self.model.load_state_dict(f"{model_checkpoint_path}")
    #     if for_eval:
    #         self.model.eval()


