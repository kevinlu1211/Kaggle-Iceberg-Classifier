import time
import torch
from pathlib import Path
import os

class TrainerDelegate(object):

    def __init__(self, experiment_save_path):
        self.experiment_save_path = experiment_save_path

        # self.training_loss = []
        # self.validation_loss = []

    def on_epoch_start(self, dataset):
        train = dataset['train']
        val = dataset['val']
        return train, val

    def create_model_input(self, data):
        return data['input']

    def create_model_output(self, model_input, model):
        model_output = model(model_input)
        return model_output

    def apply_output_transformation(self, model_output, output_transformer):
        if output_transformer is not None:
            return output_transformer(model_output)
        else:
            return model_output

    def create_data_labels(self, data):
        return data['label']

    def calculate_loss(self, loss_function, model_output, labels, mode):
        loss = loss_function(model_output, labels)
        return loss

    def apply_backward_pass(self, optimizer, model_loss, model):
        optimizer.zero_grad()
        model_loss.backward()
        optimizer.step()

    def on_finish_train(self, model, epoch):
        pass

    def on_finish_validation(self, model, epoch):
        models_checkpoint_folder_path = Path(f"{self.experiment_save_path}/saved_models/{model_id}")
        models_checkpoint_folder_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), os.fspath(f"{models_checkpoint_folder_path}/epoch-{epoch}"))

    def on_load_model(self, experiment_path, ckpt_name, for_eval):
        assert ckpt_name is not None
        model_id = self.metadata["id"]
        model_checkpoint_path = f"{experiment_path}/saved_models/{model_id}/{ckpt_name}"
        self.model.load_state_dict(f"{model_checkpoint_path}")
        if for_eval:
            self.model.eval()


