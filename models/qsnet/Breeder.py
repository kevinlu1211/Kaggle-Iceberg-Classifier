from abc import ABC, abstractmethod
import random
from uuid import uuid4
import json
from pathlib import Path
from constraint import Problem
import numpy as np
from copy import deepcopy
import pandas as pd


class Breeder(ABC):
    def __init__(self, config, experiment_factory):
        self.config = config
        self.experiment_factory = experiment_factory
        self.results = []

    @abstractmethod
    def create_experiment_config(self):
        pass


class QSNetBreeder(Breeder):
    def __init__(self, config, experiment_factory, study_save_path, training_data_path, testing_data_path):
        super().__init__(config, experiment_factory)
        self.study_save_path = study_save_path
        self.training_data_path = training_data_path
        self.testing_data_path = testing_data_path

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
        problem.addConstraint(lambda lr, decay: lr > decay, ("lr", "weight_decay"))
        solutions = problem.getSolutions()
        return {"name": optimizer,
                "parameters": random.sample(solutions, 1)[0]}

    def sample_model(self):
        model_config = dict()

        base_experiment_config = self.config['base_experiment_config']
        model_name = base_experiment_config["model"]["name"]
        model_config["n_blocks"] = 4
        prev_out_channels = int(conv[0]["parameters"]["in_channels"])
        for i in range(int(model_config['n_blocks'])):

            # Pooling
            model_config[f"block_{i}.pooling"] = random.sample(pooling, 1)[0]

            # Non-linearity
            model_config[f"block_{i}.non_linearity"] = random.sample(non_linearity, 1)[0]

            # Conv Layer
            conv_layer = deepcopy(conv[i])
            conv_layer_parameters = conv_layer["parameters"]
            model_config[f"block_{i}.conv"] = conv_layer
            n_in_channels = prev_out_channels

            # Sample out_channels
            n_out_channels = int(np.random.normal(**conv_layer_parameters["out_channels"]))
            conv_layer_parameters["in_channels"] = n_in_channels
            conv_layer_parameters["out_channels"] = n_out_channels

            # Update conv layer parameters
            model_config[f"block_{i}.conv"].update({"parameters": conv_layer_parameters})

            # Batch Norm
            model_config[f"block_{i}.batch_norm"] = {"name": "BatchNorm2d",
                                                     "parameters": {
                                                              "num_features": n_in_channels
                                                        }
                                                     }

            # Dropout
            dropout_probs = np.linspace(0, 0.5, 11).tolist()
            model_config[f"block_{i}.dropout"] = {"name": "Dropout",
                                                  "parameters": {
                                                            "dropout_rate": random.sample(dropout_probs, 1)[0]
                                                        }
                                                    }
            prev_out_channels = n_out_channels

        return {
                "name": model_name,
                "config": model_config
                }

    def create_experiment_config(self, model_config=None, optimizer_config=None):
        experiment_id = str(uuid4())
        base_experiment_config = self.config["base_experiment_config"]
        n_epochs = random.sample(base_experiment_config['n_epochs'], 1)[0]
        if model_config is None:
            model = self.sample_model()
        else:
            model = model_config
        if optimizer_config is None:
            optimizer = self.sample_optimizer()
        else:
            optimizer = optimizer_config
        loss_function = base_experiment_config['loss_function']
        output_transformation = base_experiment_config['output_transformation']
        trainer_delegate = base_experiment_config['trainer_delegate']
        data_source_delegate = base_experiment_config['data_source_delegate']
        result_delegate = base_experiment_config['result_delegate']

        config = {
            "id": experiment_id,
            "model": model,
            "optimizer": optimizer,
            "loss_function": loss_function,
            "output_transformation": output_transformation,
            "trainer_delegate": trainer_delegate,
            "data_source_delegate": data_source_delegate,
            "result_delegate": result_delegate,
            "n_epochs": n_epochs
        }
        return config

    def start_breeding(self):
        experiment_configs = [self.create_experiment_config() for _ in range(self.config['population_size'])]
        for gen in range(self.config['n_generations']):
            minion_data = []
            for experiment_config in experiment_configs:
                minion_data.append(self.raise_minion(experiment_config))
                self.record_experiment(experiment_config, gen)
            new_minions = self.breed_minions(minion_data)
            experiment_configs = [self.create_experiment_config(model_config=minion["model"],
                                                                optimizer_config=minion["optimizer"])
                                  for minion in new_minions]

    def record_experiment(self, experiment_config, generation):
        if "params" in experiment_config["optimizer"]["parameters"]:
            del experiment_config["optimizer"]["parameters"]["params"]

        experiment_id = experiment_config["id"]
        config_save_path = Path(f"{self.study_save_path}/{experiment_id}") if generation is None else \
            Path(f"{self.study_save_path}/generation_{generation}/{experiment_id}")
        config_save_path.mkdir(parents=True, exist_ok=True)

        with open(f"{config_save_path}/experiment_config.json", "w") as fp:
            json.dump(experiment_config, fp, indent=2)

    def raise_minion(self, experiment_config):
        experiment = self.experiment_factory.create_experiment(experiment_config,
                                                               self.study_save_path,
                                                               self.training_data_path,
                                                               self.testing_data_path)
        experiment.train()
        minion_data = {
            "train_results": experiment.result_delegate.training_results,
            "validation_results": experiment.result_delegate.validation_results,
            "model_config": experiment_config["model"],
            "optimizer_config": experiment_config["optimizer"]
        }
        return minion_data

    def breed_minions(self, minions_data):
        ranking_data = []
        for data in minions_data:
            rank_data = self.rank_minion(data['model_config'],
                                         data['optimizer_config'],
                                         data['validation_results'])
            ranking_data.append(rank_data)
        survived_minions = self.select_minions(ranking_data)
        new_minions = self.create_minions()
        new_and_survived_minions = survived_minions + new_minions
        mutated_minions = self.mutate_minions(new_and_survived_minions)
        all_minions = mutated_minions + new_and_survived_minions
        return all_minions


    def rank_minion(self, model_config, optimizer_config, results, k=0.75):
        rank_data = {}
        n_epochs = len(results[0])
        last_k_results = int(n_epochs * k)
        means = []
        vars = []
        for fold, result in results.items():
            losses = np.array(list(result.values())[last_k_results:])
            mean = np.mean(losses)
            var = np.var(losses)
            # rank_data[f"fold_{fold}"] = {"mean": mean,
            #                              "var": var,
            #                             }
            means.append(mean)
            vars.append(var)
        rank_data["mean"] = np.mean(means)
        rank_data["var"] = np.var(vars)
        rank_data["model"] = model_config
        rank_data["optimizer"] = optimizer_config
        return rank_data


    def select_minions(self, ranking_data):
        # TODO: Use Z-score based sampling to select minions
        minion_rank = pd.DataFrame(ranking_data)
        minion_rank.sort_values(["mean"], ascending=[True], inplace=True)
        self.results.append(minion_rank["mean"].tolist())
        new_generation_minions = deepcopy(minion_rank.head(max(1,
                                                      int(self.config['population_size'] * self.config['keep_ratio']))))
        new_generation_minions.drop(["mean", "var"], axis=1, inplace=True)

        return [{"model": v["model"], "optimizer": v["optimizer"]}
                for v in new_generation_minions.to_dict("index").values()]

    def create_minions(self):
        minions_to_create = max(1, int(self.config['population_size'] * self.config['create_ratio']))
        return [{"model": self.sample_model(), "optimizer": self.sample_optimizer()} for _ in range(minions_to_create)]

    def mutate_minion(self, minion_1, minion_2):
        minion_1_model_config = minion_1["model"]["config"]
        minion_2_model_config = minion_2["model"]["config"]
        new_minion_model_config = {}

        # TODO: Probably replace this with a dependency dictionary, so iterate through the model_config
        # TODO: which should have something like: layer: {name: "blabla", "parameters": {"p1": "bla", ...},
        # TODO: "dependencies": {"layer_1": ["p1", "p2", ...]}

        for k in minion_1_model_config.keys():
            minion_1_value = minion_1_model_config[k]
            minion_2_value = minion_2_model_config[k]
            if "Conv2d" in k or "BatchNorm2d":
                new_minion_model_config[k] = minion_1_value
            else:
                new_minion_model_config[k] = random.sample([minion_1_value, minion_2_value], 1)[0]

        new_minion_optimizer_config = random.sample([minion_1["optimizer"],  minion_2["optimizer"]], 1)[0]

        return {"model": {"name": minion_1["model"]["name"], "config": new_minion_model_config},
                "optimizer": new_minion_optimizer_config}

    def mutate_minions(self, minions):
        mutated_minions = []
        minions_to_mutate = max(1, int(self.config['population_size'] * self.config['mutate_ratio']))
        for _ in range(minions_to_mutate):
            mutated_minion = self.mutate_minion(*random.sample(minions, 2))
            mutated_minions.append(mutated_minion)

        return mutated_minions







### REFACTOR THIS OUT INTO A CONFIG FILE ###

non_linearity = [
          {
            "name": "LeakyReLU",
            "parameters": {
              "inplace": True
            }
          },
          {
            "name": "ReLU",
            "parameters": {
              "inplace": True
            }
          }
        ]

pooling = [
        {
          "name": "MaxPool2d",
          "parameters": {
            "kernel_size": 2,
            "stride": 2
          }
        },
        {
          "name": "AvgPool2d",
          "parameters": {
            "kernel_size": 2,
            "stride": 2
          }
        }
      ]

conv = [
        {
          "name": "Conv2d",
          "parameters": {
            "in_channels": 3,
            "out_channels": {
              "loc": 25,
              "scale": 5
            },
            "kernel_size": 5,
            "stride": 1
          }
        },
        {
          "name": "Conv2d",
          "parameters": {
            "in_channels": None,
            "out_channels": {
              "loc": 50,
              "scale": 10
            },
            "kernel_size": 3,
            "stride": 1
          }
        },
        {
          "name": "Conv2d",
          "parameters": {
            "in_channels": None,
            "out_channels": {
              "loc": 100,
              "scale": 10
            },
            "kernel_size": 3,
            "stride": 1
          }
        },
        {
          "name": "Conv2d",
          "parameters": {
            "in_channels": None,
            "out_channels": {
              "loc": 200,
              "scale": 20
            },
            "kernel_size": 3,
            "stride": 1
          }
        }
]








