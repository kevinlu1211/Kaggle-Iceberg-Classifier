import argparse
import logging
from src.ExperimentMappings \
    import model_lookup, \
    loss_function_lookup, \
    optimizer_lookup, \
    scheduler_lookup, \
    data_source_delegates_lookup, \
    trainer_delegates_lookup, \
    evaluation_delegates_lookup, \
    saver_delegates_lookup
from src.Experiment import ExperimentFactory
import json
import itertools
import numpy as np
import random
import torch

def main():
    with open(opts.experiment_config_path, "r") as fp:
        experiment_config = json.load(fp)
    fix_seed(experiment_config.get("seed"))
    experiment_factory = ExperimentFactory(model_lookup,
                                           loss_function_lookup,
                                           optimizer_lookup,
                                           scheduler_lookup,
                                           data_source_delegates_lookup,
                                           trainer_delegates_lookup,
                                           evaluation_delegates_lookup,
                                           saver_delegates_lookup)

    # TODO: think of a better way to do this, probably have a separate object to create the config files
    # hyperparameters = []
    lrs = experiment_config["optimizer"]["parameters"]["lr"]
    # hyperparameters.append(lrs)
    try:
        weight_decays = experiment_config["optimizer"]["parameters"]["weight_decay"]
        dropouts = experiment_config["model"]["parameters"]["dropout_rates"]
        for lr, weight_decay, dropout in itertools.product(*[lrs, weight_decays, dropouts]):
            new_experiment_config = experiment_config.copy()
            new_experiment_config["optimizer"]["parameters"]["lr"] = lr
            new_experiment_config["optimizer"]["parameters"]["weight_decay"] = weight_decay
            new_experiment_config["model"]["parameters"]["dropout_rates"] = dropout
            experiment = experiment_factory.create_experiment(new_experiment_config, opts.study_save_path)
            experiment.train()
    except KeyError:
        weight_decays = experiment_config["optimizer"]["parameters"]["weight_decay"]
        for lr, weight_decay in itertools.product(*[lrs, weight_decays]):
            new_experiment_config = experiment_config.copy()
            new_experiment_config["optimizer"]["parameters"]["lr"] = lr
            new_experiment_config["optimizer"]["parameters"]["weight_decay"] = weight_decay
            experiment = experiment_factory.create_experiment(new_experiment_config, opts.study_save_path)
            experiment.train()


    # Fine tuning
    # lrs = experiment_config["optimizer"]["parameters"]["lr"]
    # ft_lrs = experiment_config["optimizer"].get("fine_tuning_parameters", dict()).get("lr")
    # for base_lr, ft_lr in itertools.product(*[lrs, ft_lrs]):
    #     new_experiment_config = experiment_config.copy()
    #     new_experiment_config["optimizer"]["parameters"]["lr"] = base_lr
    #     new_experiment_config["optimizer"]["fine_tuning_parameters"]["lr"] = ft_lr
        experiment = experiment_factory.create_experiment(new_experiment_config, opts.study_save_path)
        # experiment.train()

def fix_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_save_path", default="../study_results/iceresnet_experiment_dataaug_h_v_flip")
    parser.add_argument("--experiment_config_path", default="study_configs/iceresnet_experiment.json")
    # parser.add_argument("--experiment_config_path", default="study_configs/triple_column_iceresnet_experiment.json")
    # parser.add_argument("--experiment_config_path", default="study_configs/iceresnet2_experiment.json")
    parser.add_argument("--logging_level", default="INFO")
    opts = parser.parse_args()
    log_level = logging.getLevelName(opts.logging_level)
    logging.basicConfig(level=log_level)
    main()
