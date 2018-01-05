import argparse
import logging
import time
from src.ExperimentMappings \
    import model_lookup, \
    loss_function_lookup, \
    optimizer_lookup, \
    data_source_delegates_lookup, \
    trainer_delegates_lookup, \
    evaluation_delegates_lookup, \
    saver_delegates_lookup
from src.Experiment import ExperimentFactory
import json


def main():
    with open(opts.experiment_config_path, "r") as fp:
        experiment_config = json.load(fp)

    experiment_factory = ExperimentFactory(model_lookup,
                                           loss_function_lookup,
                                           optimizer_lookup,
                                           data_source_delegates_lookup,
                                           trainer_delegates_lookup,
                                           evaluation_delegates_lookup,
                                           saver_delegates_lookup)
    experiment = experiment_factory.create_experiment(experiment_config, opts.study_save_path)
    experiment.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_save_path", default="../study_results/densenet")
    parser.add_argument("--experiment_config_path", default="../study_configs/densenet_experiment.json")
    parser.add_argument("--logging_level", default="INFO")
    opts = parser.parse_args()
    log_level = logging.getLevelName(opts.logging_level)
    logging.basicConfig(level=log_level)
    main()
