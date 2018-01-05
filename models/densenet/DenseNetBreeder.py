from scipy.stats import geom
import random
from uuid import uuid4
import json
from pathlib import Path
from constraint import Problem
import numpy as np
from copy import deepcopy
import pandas as pd
from src.Breeder import AbstractBreeder
from collections import OrderedDict


class DenseNetModelConfigurationCreator(object):
    def __init__(self, config):
        self.config = config

    def sample_optimizer(self):
        base_experiment_config = self.config["base_experiment_config"]
        optimizers_and_parameters = list(zip(base_experiment_config["optimizer"]["name"],
                                             base_experiment_config["optimizer"]["parameters"]))
        sampled = random.sample(optimizers_and_parameters, 1)[0]
        optimizer = sampled[0]
        parameters = sampled[1]
        problem = Problem()
        for k, v in parameters.items():
            problem.addVariable(k, v)
        # problem.addConstraint(lambda lr, decay: lr > decay, ("lr", "weight_decay"))
        solutions = problem.getSolutions()
        return {"name": optimizer,
                "parameters": random.sample(solutions, 1)[0]}

    def sample_dropout(self):
        return random.sample(self.config["model_mutations"]["dropout_range"], 1)[0]

    def sample_pooling(self):
        return random.sample(self.config["model_mutations"]["pooling"], 1)[0]

    def sample_non_linearity(self):
        return random.sample(self.config["model_mutations"]["non_linearity"], 1)[0]

    def sample_model_config(self):
        model_config = dict()
        base_experiment_config = self.config['base_experiment_config']
        base_model_config = base_experiment_config["model"]["config"]

        model_name = base_experiment_config["model"]["name"]
        model_config.update(**base_model_config)

        layers_per_block = base_model_config['layers_per_block']
        growth_rate = base_model_config['growth_rate']
        bottleneck_size = base_model_config['bottleneck_size']
        n_features = base_model_config['n_initial_features']

        initial_block = self.make_initial_block_config(n_features)
        model_config.update(**initial_block)

        for block_i, n_layers in enumerate(layers_per_block):
            block_config = self.make_dense_block_config(block_i, n_layers, n_features, growth_rate, bottleneck_size)
            model_config.update(**block_config)
            n_features = n_features + growth_rate * n_layers
            if block_i != len(block_config) - 1:
                trans = self.make_transition_layer_config(block_i, n_input_features=n_features,
                                                     n_output_features=n_features // 2)
                n_features = n_features // 2
                model_config.update(**trans)

        return {
                "name": model_name,
                "config": model_config
                }


    def make_initial_block_config(self, n_init_features, bias=False):
        prefix = "initial_block"
        block_config = dict()
        block_config[f"{prefix}.conv_0"] = {
            "name": "Conv2d",
            "parameters": {
                "in_channels": 3,
                "out_channels": n_init_features,
                "kernel_size": 5,
                "stride": 1,
                "bias": bias
            }
        }
        block_config[f"{prefix}.batch_norm_0"] = {
            "name": "BatchNorm2d",
            "parameters": {
                "num_features": n_init_features
            }
        }
        block_config[f"{prefix}.non_linearity_0"] = self.sample_non_linearity()
        # block_config[f"{prefix}.pooling_0"] = {
        #     "name": "AvgPool2d",
        #     "parameters": {
        #         "kernel_size": 2,
        #         "stride": 2
        #     }
        # }
        return block_config

    def make_dense_block_config(self, block_i, n_layers, n_input_features, growth_rate, bottleneck_size,
                                bias=False):
        block_config = OrderedDict()
        for layer_i in range(n_layers):
            layer_config = self.make_dense_layer_config(block_i, layer_i, n_input_features + layer_i * growth_rate,
                                                   growth_rate,
                                                   bottleneck_size, bias)
            block_config.update(**layer_config)
        return block_config

    def make_dense_layer_config(self, block_i, layer_i, n_input_features, growth_rate, bottleneck_size,
                                bias=False):
        layer_config = OrderedDict()
        prefix = f"denseblock_{block_i}.layer_{layer_i}"
        layer_config[f"{prefix}.batch_norm_0"] = {
            "name": "BatchNorm2d",
            "parameters": {
                "num_features": n_input_features
            }
        }
        layer_config[f"{prefix}.non_linearity_0"] = self.sample_non_linearity()
        layer_config[f"{prefix}.conv_0"] = {
            "name": "Conv2d",
            "parameters": {
                "in_channels": n_input_features,
                "out_channels": growth_rate * bottleneck_size,
                "kernel_size": 1,
                "stride": 1,
                "bias": bias
            }
        }
        layer_config[f"{prefix}.dropout_0"] = {
            "name": "Dropout",
            "parameters": {
                "p": self.sample_dropout()
            }
        }
        layer_config[f"{prefix}.batch_norm_1"] = {
            "name": "BatchNorm2d",
            "parameters": {
                "num_features": growth_rate * bottleneck_size
            }
        }
        layer_config[f"{prefix}.non_linearity_1"] = self.sample_non_linearity()
        layer_config[f"{prefix}.conv_1"] = {
            "name": "Conv2d",
            "parameters": {
                "in_channels": growth_rate * bottleneck_size,
                "out_channels": growth_rate,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "bias": bias
            }
        }
        layer_config[f"{prefix}.dropout_1"] = {
            "name": "Dropout",
            "parameters": {
                "p": self.sample_dropout()
            }
        }
        return layer_config

    def make_transition_layer_config(self, block_i, n_input_features, n_output_features, bias=False):
        prefix = f"transition_{block_i}"
        layer_config = OrderedDict()
        layer_config[f"{prefix}.batch_norm_0"] = {
            "name": "BatchNorm2d",
            "parameters": {
                "num_features": n_input_features
            }
        }
        layer_config[f"{prefix}.non_linearity_0"] = self.sample_non_linearity()
        layer_config[f"{prefix}.conv_0"] = {
            "name": "Conv2d",
            "parameters": {
                "in_channels": n_input_features,
                "out_channels": n_output_features,
                "kernel_size": 1,
                "stride": 1,
                "padding": 1,
                "bias": bias
            }
        }
        layer_config[f"{prefix}.pooling_0"] = self.sample_pooling()
        layer_config[f"{prefix}.dropout_0"] = {
            "name": "Dropout",
            "parameters": {
                "p": self.sample_dropout()
            }
        }
        return layer_config


    def create_experiment_config(self, model_config=None, optimizer_config=None):
        experiment_id = str(uuid4())
        base_experiment_config = self.config["base_experiment_config"]
        n_epochs = random.sample(base_experiment_config['n_epochs'], 1)[0]
        if model_config is None:
            model = self.sample_model_config()
        else:
            model = model_config
        if optimizer_config is None:
            optimizer = self.sample_optimizer()
        else:
            optimizer = optimizer_config

        scheduler = base_experiment_config.get('scheduler', dict())
        loss_function = base_experiment_config['loss_function']
        output_transformation = base_experiment_config['output_transformation']
        trainer_delegate = base_experiment_config['trainer_delegate']
        data_source_delegate = base_experiment_config['data_source_delegate']
        result_delegate = base_experiment_config['result_delegate']
        saver_delegate = base_experiment_config['saver_delegate']



        config = {
            "id": experiment_id,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "loss_function": loss_function,
            "output_transformation": output_transformation,
            "trainer_delegate": trainer_delegate,
            "data_source_delegate": data_source_delegate,
            "result_delegate": result_delegate,
            "saver_delegate": saver_delegate,
            "n_epochs": n_epochs
        }
        return config


class DenseNetBreeder(AbstractBreeder):
    def __init__(self, config, experiment_factory, study_save_path):
        super().__init__(config, experiment_factory)
        self.study_save_path = study_save_path

        # TODO: Need to do some thinking about how to make the architecture nicer, should I use
        # TODO: another dependency injection?
        self.model_config_creator = DenseNetModelConfigurationCreator(config)

    def start_breeding(self):
        experiment_configs = [self.model_config_creator.create_experiment_config()
                              for _ in range(self.config['population_size'])]
        for generation in range(self.config['n_generations']):
            minion_data = []
            for experiment_config in experiment_configs:
                minion = self.raise_minion(experiment_config, generation)
                minion_data.append(minion)
                self.record_experiment(minion["train_results"],
                                       minion["validation_results"],
                                       minion["generation"],
                                       experiment_config)
            new_minions = self.breed_minions(minion_data)
            experiment_configs = [self.model_config_creator.create_experiment_config(
                                                                model_config=minion["model"],
                                                                optimizer_config=minion["optimizer"])
                                  for minion in new_minions]

    def record_experiment(self, train_results, val_results, generation, experiment_config):

        # First delete the objects from the dictionary so that it can be serialized
        try:
            del experiment_config["optimizer"]["parameters"]["params"]
        except KeyError:
            pass
        try:
            del experiment_config["scheduler"]["parameters"]["optimizer"]
        except KeyError:
            pass

        experiment_id = experiment_config["id"]
        experiment_config["generation"] = generation
        save_path = Path(f"{self.study_save_path}/generation_{generation}/{experiment_id}")
        save_path.mkdir(parents=True, exist_ok=True)

        with open(f"{save_path}/experiment_config.json", "w") as fp:
            json.dump(experiment_config, fp, indent=2)

        results = self.create_rank_results(train_results, val_results)
        for k, v in results.items():
            if isinstance(v, np.float32):
                results[k] = float(v)

        with open(f"{save_path}/experiment_results.json", "w") as fp:
                json.dump(results, fp, indent=2)

    def raise_minion(self, experiment_config, generation):
        experiment = self.experiment_factory.create_experiment(experiment_config,
                                                               self.study_save_path,
                                                               generation)
        experiment.train()
        minion_data = {
            "train_results": experiment.saver_delegate.training_results,
            "validation_results": experiment.saver_delegate.validation_results,
            "model_config": experiment_config["model"],
            "optimizer_config": experiment_config["optimizer"],
            "generation": generation
        }
        return minion_data

    def breed_minions(self, minions_data):
        ranking_data = []
        for data in minions_data:
            rank_data = self.create_rank_results(data['train_results'],
                                                 data['validation_results'])
            rank_data["model"] = data['model_config']
            rank_data["optimizer"] = data['optimizer_config']
            ranking_data.append(rank_data)

        survived_minions = self.select_minions(ranking_data)
        new_minions = self.create_minions()
        new_and_survived_minions = survived_minions + new_minions
        mutated_minions = self.mutate_minions(new_and_survived_minions)
        all_minions = mutated_minions + new_and_survived_minions
        return all_minions

    # TODO: Make an interface so that we can have different selection strategies
    def create_rank_results(self, train_results, val_results, k=0.75):
        rank_data = {}
        n_epochs = len(train_results[0])
        last_k_results = int(n_epochs * k)
        means = []
        vars = []
        loss_diffs = []
        for (_, train_result), (_, val_result) in zip(train_results.items(), val_results.items()):
            train_losses = np.array(list(train_result.values())[last_k_results:])
            val_losses = np.array(list(val_result.values())[last_k_results:])
            train_val_loss_diff = np.mean(train_losses - val_losses)
            mean = np.mean(val_losses)
            var = np.var(val_losses)
            loss_diffs.append(train_val_loss_diff)
            means.append(mean)
            vars.append(var)
        rank_data["val_loss_mean"] = np.mean(means)
        rank_data["val_loss_var"] = np.var(vars)
        rank_data["train_val_loss_avg_diff"] = np.mean(loss_diffs)
        return rank_data

    def select_minions(self, ranking_data):
        minion_rank = pd.DataFrame(ranking_data)
        n_to_select = int(self.config['population_size'] * self.config['keep_ratio'])

        minion_rank.sort_values(["train_val_loss_avg_diff"], ascending=[False], inplace=True)
        lowest_train_val_diff_loss = deepcopy(minion_rank.head(max(1, n_to_select//2 + 1)).reset_index(drop=True))

        minion_rank.sort_values(["val_loss_mean"], ascending=[True], inplace=True)
        lowest_mean_loss = deepcopy(minion_rank.head(max(1, n_to_select//2 + 1)).reset_index(drop=True))
        new_generation_minions = pd.concat([lowest_train_val_diff_loss, lowest_mean_loss])
        new_generation_minions.sort_index(inplace=True)
        new_generation_minions.drop(["val_loss_mean", "val_loss_var", "train_val_loss_avg_diff"], axis=1, inplace=True)

        return [{"model": v["model"], "optimizer": v["optimizer"]}
                for v in new_generation_minions.to_dict("index").values()]

    def create_minions(self):
        minions_to_create = max(1, int(self.config['population_size'] * self.config['create_ratio']))
        return [{"model": self.model_config_creator.sample_model_config(),
                 "optimizer": self.model_config_creator.sample_optimizer()}
                for _ in range(minions_to_create)]

    def mutate_minion(self, minion_1, minion_2):
        minion_1_model_config = minion_1["model"]["config"]
        minion_2_model_config = minion_2["model"]["config"]
        new_minion_model_config = {}

        # TODO: Probably replace this with a dependency dictionary, so iterate through the model_config
        # TODO: which should have something like:
        # TODO:     layer: {name: "blabla",
        # TODO:                   "parameters": {"p1": "bla", ...},
        # TODO:                   "dependencies": {"layer_1": ["p1", "p2", ...]
        # TODO:            }

        for k in minion_1_model_config.keys():
            minion_1_value = minion_1_model_config[k]
            minion_2_value = minion_2_model_config[k]
            if "Conv2d" in k or "BatchNorm2d" in k:
                new_minion_model_config[k] = minion_1_value
            else:
                new_minion_model_config[k] = random.sample([minion_1_value, minion_2_value], 1)[0]

        new_minion_optimizer_config = random.sample([minion_1["optimizer"],  minion_2["optimizer"]], 1)[0]

        return {"model": {
                    "name": minion_1["model"]["name"],
                    "config": new_minion_model_config
                    },
                "optimizer": new_minion_optimizer_config
                }

    def sample_minions(self, minions):
        p = geom.pmf(range(1, len(minions) + 1), p=0.1)
        weights = p/sum(p)
        leftover = 1 - sum(p)
        p += weights*leftover
        minions = np.random.choice(minions, size=2, p=p)
        return minions

    def mutate_minions(self, minions):
        mutated_minions = []
        minions_to_mutate = max(1, int(self.config['population_size'] * self.config['mutate_ratio']))
        for _ in range(minions_to_mutate):
            sampled_minions = self.sample_minions(minions)
            mutated_minion = self.mutate_minion(*sampled_minions)
            mutated_minions.append(mutated_minion)

        return mutated_minions

