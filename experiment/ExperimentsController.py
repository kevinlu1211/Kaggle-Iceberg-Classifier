import json
import os
import torch
from time import strftime
import logging
from pathlib import Path


class ExperimentsController(object):

    create_component = lambda x, y: x(**y) if y else x()

    def __init__(self, experiment_factory):
        self.experiment_factory = experiment_factory

    def setup_experiments(self, experiment_path, metadata_file_paths):
        self.experiment_path = experiment_path
        self.experiments_metadata = self.load_experiments_metadata(metadata_file_paths)
        self.experiments = [self.experiment_factory.create_experiment(exp_metadata)
                            for exp_metadata in self.experiments_metadata]

        # Dump the models metadata into the /models_metadata folder which conveniently creates
        # the folder to hold all the data for this experiment
        self.copy_metadata_files()
        self.experiment_is_created = True

    def load_experiments_metadata(self, metadata_file_paths):
        models_metadata = []
        for path in metadata_file_paths:
            with open(path, "r") as fp:
                model_metadata = json.load(fp)
                models_metadata.append(model_metadata)
        return models_metadata

    def copy_metadata_files(self):
        experiments_metadata_path = Path(f"{self.experiment_path}/experiments_metadata")
        experiments_metadata_path.mkdir(parents=True, exist_ok=True)
        for experiment_metadata, metadata_file_path in zip(self.experiments_metadata, self.metadata_file_paths):
            file_name = metadata_file_path.split("/")[-1]
            with open(f"{experiments_metadata_path}/{file_name}", 'w') as fp:
                json.dump(experiment_metadata, fp)

    def save_experiments(self, ckpt_names=None):
        if ckpt_names is None:
            ckpt_names = [strftime("%Y%m%d-%H%M%S")] * len(self.models)

        for ckpt_name, experiment in zip(ckpt_names, self.experiments):
            experiment.save_experiment(self.experiment_path, ckpt_name)

    def load_experiments(self, experiment_path, metadata_file_paths, ckpt_names, for_eval=True):
        self.experiments_metadata = self.load_experiments_metadata(metadata_file_paths)
        self.experiments = [self.experiment_factory.create_experiment(exp_metadata)
                            for exp_metadata in self.experiments_metadata]
        for ckpt_name, experiment in zip(ckpt_names, self.experiments):
            experiment.load_experiment(experiment_path, ckpt_name, for_eval)

    def train_in_parallel(self, inp, label):
        all_experiment_outputs = {}
        all_experiment_losses = {}
        for experiment in self.experiments:
            output, loss = experiment.train_step(inp, label)
            all_experiment_outputs[experiment.name] = output
            all_experiment_losses[experiment.name] = loss

        return all_experiment_outputs, all_experiment_losses

    def train_in_sequence(self, all_inps, all_labels):
        pass








