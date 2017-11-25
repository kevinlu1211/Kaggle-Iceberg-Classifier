import json
import os
import torch
from time import strftime
from nn.autograd import Variable
import logging
from pathlib import Path


class Experiment(object):
    def __init__(self, model_ptrs, model_preprocessing_ptrs, optimizer_ptrs, loss_function_ptrs):
        self.model_ptrs = model_ptrs
        self.model_preprocessing_ptrs = model_preprocessing_ptrs
        self.optimizer_ptrs = optimizer_ptrs
        self.loss_function_ptrs = loss_function_ptrs


    def setup_experiment(self, experiment_path, metadata_file_paths):
        self.models_metadata = self.load_models_metadata(metadata_file_paths)
        self.models = self.build_models()
        self.optimizers = self.build_optimizers()
        self.loss_functions = self.build_loss_functions()
        self.metadata_file_paths = metadata_file_paths
        self.experiment_path = experiment_path

        # Dump the models metadata into the /models_metadata folder which conveniently creates
        # the folder to hold all the data for this experiment
        models_metadata_path = Path(f"{self.experiment_path}/models_metadata")
        models_metadata_path.mkdir(parents=True, exist_ok=True)
        self.experiment_is_created = True

    def save_experiment(self, ckpt_names=None):

        if ckpt_names is None:
            ckpt_names = [strftime("%Y%m%d-%H%M%S")] * len(self.models)

        if self.experiment_is_created:
            for model_metadata, model, ckpt_name in zip(self.models_metadata, self.models, ckpt_names):
                model_id = model_metadata["id"]
                models_checkpoint_path = Path(f"{self.experiment_path}/saved_models/{model_id}")
                models_checkpoint_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), os.fspath(f"{models_checkpoint_path}/{ckpt_name}"))

    def load_experiment(self, experiment_path, metadata_file_paths, ckpt_names, for_eval=True):
        """

        :param experiment_path:
        :param metadata_file_paths:
        :param ckpt_names: This has to be constructed such that the name of the checkpoints are in the same order that
                           the metadata_file_paths are in.

                           For example:

                           /path/to/experiment/
                               models_metadata/
                                   model_1.meta
                                   model_2.meta
                               saved_models/
                                   model_1_id/
                                       checkpoint_1.pth
                                       checkpoint_2.pth
                                   model_2_id/
                                       checkpoint_3.pth
                                       checkpoint_4.pth

                           then if we want checkpoint_2 from model_1, and checkpoint_3 from model_2, we would have to
                           pass in:

                           ckpt_names = ["checkpoint_2.pth", "checkpoint_3.pth"]
        :param for_eval:
        :return:
        """
        if ckpt_names == None:
            logging.error("Please provide checkpoint names")
            return
        self.models_metadata = self.load_models_metadata(metadata_file_paths)
        self.models = self.build_models()
        self.optimizers = self.build_optimizers()
        self.experiment_path = experiment_path
        for model_metadata, model in zip(self.models_metadata, self.models):
            model_id = model_metadata["id"]
            model.load_state_dict(f"{experiment_path}/{model_id}/{ckpt_name}")
            if for_eval:
                model.eval()

    def load_model_metadata(self, meta_data_file_paths):
        return [json.loads(p) for p in meta_data_file_paths]


    # TODO: should I refactor build_{model, optimizer, loss} into build_component?

    def build_models(self):
        models = []
        for model_metadata in self.models_metadata:
            models.append(self.build_model(model_metadata))
        return models

    def build_model(self, model_metadata):
        model_name = model_metadata["model_name"]
        model_parameters = model_metadata["model_parameters"]
        return self.model_ptrs[model_name](**model_parameters)

    def build_optimizers(self):
        optimizers = []
        for model_metadata in self.models_metadata:
            optimizers.append(self.build_optimizer(model_metadata))
        return optimizers

    def build_optimizer(self, model_metadata):
        optimizer_name = model_metadata["optimizer_name"]
        optimizer_parameters = model_metadata["optimizer_parameters"]
        return self.optimizer_ptrs[optimizer_name](**optimizer_parameters)

    def build_loss_functions(self):
        loss_functions = []
        for model_metadata in self.models_metadata:
            loss_functions.append(self.build_loss_function(model_metadata))
        return loss_functions

    def build_loss_function(self, model_metadata):
        loss_function_name = model_metadata["loss_function_name"]
        loss_function_parameters = model_metadata["loss_function_parameters"]
        return self.loss_function_ptrs[loss_function_name](**loss_function_parameters)

    def get_model_specific_preprocessor(self, model_name, preprocessing_method_name):
        try:
            return self.model_preprocessing_ptrs[model_name][preprocessing_method_name]
        except KeyError as ke:
            logging.error(ke)
            return None

    def get_loss_function(self, loss_function_name):
        return self.loss_function_ptrs[loss_function_name]

    def train_step(self, inp_data, label, output_transformation=None):
        all_model_outputs = {}
        all_model_losses = {}
        for model_metadata, model, optimizer, loss_function in zip(self.models_metadata,
                                                                   self.models,
                                                                   self.optimizers,
                                                                   self.loss_functions):
            model_name = model_metadata['model_name']
            preprocessing_method_name = model_metadata['preprocessing_method_name']
            preprocessing_fn = self.get_model_specific_preprocessor(model_name, preprocessing_method_name)
            preprocessed_inp = Variable(preprocessing_fn(inp_data))
            model_output = model(preprocessed_inp)

            # Transform the output here as we will get unnormalized probailities from our models
            # since we don't want our models to be coupled with the data
            transformed_output = output_transformation(model_output)
            model_loss = loss_function(transformed_output, label)

            all_model_outputs[model_name] = transformed_output
            all_model_losses[model_name] = model_loss

        return all_model_outputs, all_model_losses








