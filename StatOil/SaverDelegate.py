import numpy as np
import pandas as pd
import logging
import os
import torch
import json
from pathlib import Path
from collections import defaultdict, OrderedDict

np.seterr(all='raise')

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
        self.best_validation_df = None
        self.best_validation_score = 1

    def save_results(self, data, model_output, model_loss, mode):
        labels = data['label'].numpy()
        ids = data['id']
        model_output = model_output.clamp(min=1e-6, max=0.999999)
        per_data_loss = -(labels * np.log(model_output.data) + (1 - labels) * np.log(1 - model_output.data)).numpy()
        per_data_loss = np.reshape(per_data_loss, -1)
        labels = np.reshape(labels, -1)
        predicted = (model_output.data > 0.5).float().cpu().squeeze().numpy()

        for id, output, label, predicted, loss in zip(ids, model_output.data.squeeze(), labels, predicted, per_data_loss):
            if mode == "TRAIN":
                self.training_results_for_epoch.append(
                    dict(id=id, output=output, label=label, predicted=predicted, loss=loss, train_or_validation=mode))
            if mode == "VALIDATION":
                self.validation_results_for_epoch.append(
                    dict(id=id, output=output, label=label, predicted=predicted, loss=loss, train_or_validation=mode))

    def _save_model(self, model, epoch, fold_num, loss):
        models_checkpoint_folder_path = Path(
            f"{self.study_save_path}/{self.experiment_id}/model_checkpoints/fold_{fold_num}")
        models_checkpoint_folder_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), os.fspath(f"{models_checkpoint_folder_path}/epoch_{epoch}-loss_{loss}.pth"))

    def update_all_results(self, epoch, fold_num):
        [d.update({"fold": fold_num, "epoch": epoch}) for d in self.training_results_for_epoch]
        self.all_results.extend(self.training_results_for_epoch)

        [d.update({"fold": fold_num, "epoch": epoch}) for d in self.validation_results_for_epoch]
        self.all_results.extend(self.validation_results_for_epoch)

    def update_loss_results(self, epoch, fold_num):
        train_df = pd.DataFrame(self.training_results_for_epoch)
        validation_df = pd.DataFrame(self.validation_results_for_epoch)
        avg_train_loss = train_df.mean()['loss']
        avg_validation_loss = validation_df.mean()['loss']
        if avg_validation_loss < self.best_validation_score:
            self.best_validation_df = validation_df
        avg_train_acc = (train_df['predicted'] == train_df['label']).mean()
        avg_validation_acc = (validation_df['predicted'] == validation_df['label']).mean()
        logging.info(f"Training loss for epoch: {epoch} is {avg_train_loss}")
        logging.info(f"Validation loss for epoch: {epoch} is {avg_validation_loss}")
        logging.info(f"Training accuracy for epoch: {epoch} is {avg_train_acc}")
        logging.info(f"Validation accuracy for epoch: {epoch} is {avg_validation_acc}")

        self.all_loss_results.append(dict(fold=fold_num, epoch=epoch,
                                          training_loss=avg_train_loss, validation_loss=avg_validation_loss))

    def on_epoch_end(self, model, epoch, fold_num):
        self.update_all_results(epoch, fold_num)
        self.update_loss_results(epoch, fold_num)
        avg_validation_loss = self.all_loss_results[-1]["validation_loss"]
        self._save_model(model, epoch, fold_num, avg_validation_loss)
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




