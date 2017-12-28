from .AbstractResultDelegate import AbstractResultDelegate
import numpy as np
import logging
import os
import torch
from pathlib import Path


class ResultDelegate(AbstractResultDelegate):

    def __init__(self, experiment_id, study_save_path):
        super().__init__()
        self.experiment_id = experiment_id
        self.study_save_path = study_save_path

    def on_model_loss(self, loss, mode):
        if mode == "TRAIN":
            self.training_loss_for_epoch.append(loss.data)
        else:
            self.validation_loss_for_epoch.append(loss.data)

    def _save_model(self, model, epoch, fold_num):
        models_checkpoint_folder_path = Path(
            f"{self.study_save_path}/{self.experiment_id}/model_checkpoints/fold_{fold_num}")
        models_checkpoint_folder_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), os.fspath(f"{models_checkpoint_folder_path}/epoch_{epoch}.pth"))

    def on_epoch_end(self, model, epoch, fold_num):
        avg_loss, = np.mean(np.array(self.training_loss_for_epoch)).cpu().numpy()
        self.validation_results[fold_num].update({epoch: avg_loss})
        logging.info(f"Training loss for epoch: {epoch} is {avg_loss}")

        avg_loss, = np.mean(np.array(self.validation_loss_for_epoch)).cpu().numpy()
        self.validation_results[fold_num].update({epoch: avg_loss})
        logging.info(f"Validation loss for epoch: {epoch} is {avg_loss}")
        self._save_model(model, epoch, fold_num)

        self.training_loss_for_epoch = []
        self.validation_loss_for_epoch = []

