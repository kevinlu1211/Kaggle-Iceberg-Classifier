import numpy as np
import pandas as pd
import logging
import os
import torch
import json
from pathlib import Path
from collections import defaultdict, OrderedDict


class SaverDelegate(object):

    def __init__(self, experiment_id, study_save_path, experiment_config):
        super().__init__()
        self.experiment_id = experiment_id
        self.study_save_path = study_save_path
        self.experiment_config = experiment_config
        self.last_epoch_validation_loss = None
        self.all_results = []
        self.all_loss_results = []
        self.training_results_for_epoch = []
        self.validation_results_for_epoch = []
        self.training_loss_for_epoch = []
        self.validation_loss_for_epoch = []

    def save_results(self, data, model_output, model_loss, mode):
        labels = data['label'].numpy()
        ids = data['id']
        try:
            per_data_loss = -(labels * np.log(model_output.data) + (1 - labels) * np.log(1 - model_output.data)).numpy()
            per_data_loss = np.reshape(per_data_loss, -1)
        except RuntimeWarning:
            print("what?")
        labels = np.reshape(labels, -1)

        for id, output, label, loss in zip(ids, model_output.data.squeeze(), labels, per_data_loss):
            self.training_results_for_epoch.append(
                dict(id=id, output=output, label=label, loss=loss, train_or_validation=mode))

        if mode == "TRAIN":
            self.training_loss_for_epoch.append(model_loss.data[0])
        else:
            self.validation_loss_for_epoch.append(model_loss.data[0])

    def _save_model(self, model, epoch, fold_num, loss):
        models_checkpoint_folder_path = Path(
            f"{self.study_save_path}/{self.experiment_id}/model_checkpoints/fold_{fold_num}")
        models_checkpoint_folder_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), os.fspath(f"{models_checkpoint_folder_path}/epoch_{epoch}-loss_{loss}.pth"))

    def on_epoch_end(self, model, epoch, fold_num):

        [d.update({"fold": fold_num, "epoch": epoch}) for d in self.training_results_for_epoch]
        self.all_results.extend(self.training_results_for_epoch)

        [d.update({"fold": fold_num, "epoch": epoch}) for d in self.validation_results_for_epoch]
        self.all_results.extend(self.validation_results_for_epoch)

        avg_train_loss = np.mean(self.training_loss_for_epoch)
        avg_validation_loss = np.mean(self.validation_loss_for_epoch)

        self.all_loss_results.append(dict(fold=fold_num, epoch=epoch,
                                          training_loss=avg_train_loss, validation_loss=avg_validation_loss))

        self._save_model(model, epoch, fold_num, avg_validation_loss)

        logging.info(f"Training loss for epoch: {epoch} is {avg_train_loss}")
        logging.info(f"Validation loss for epoch: {epoch} is {avg_validation_loss}")
        self.last_epoch_validation_loss = avg_validation_loss
        self.training_results_for_epoch = []
        self.validation_results_for_epoch = []
        self.training_loss_for_epoch = []
        self.validation_loss_for_epoch = []

    def on_end_experiment(self):
        experiment_save_path = Path(
            f"{self.study_save_path}/{self.experiment_id}"
        )
        experiment_save_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.all_results).to_csv(f"{experiment_save_path}/all_results.csv", index=False)
        pd.DataFrame(self.all_loss_results).to_csv(f"{experiment_save_path}/loss_results.csv", index=False)

        try:
            del self.experiment_config["optimizer"]["parameters"]["params"]
        except KeyError:
            pass
        try:
            del self.experiment_config["scheduler"]["parameters"]["optimizer"]
        except KeyError:
            pass
        with open(f"{experiment_save_path}/experiment_config.json", "w") as fp:
            json.dump(self.experiment_config, fp, indent=2)




