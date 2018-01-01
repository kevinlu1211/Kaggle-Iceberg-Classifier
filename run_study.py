import argparse
import logging
import time
from src.ExperimentMappings \
    import model_lookup, \
    loss_function_lookup, \
    optimizer_lookup, \
    output_transformation_lookup, \
    data_source_delegates_lookup, \
    trainer_delegates_lookup, \
    evaluation_delegates_lookup, \
    saver_delegates_lookup
from src.Experiment import ExperimentFactory
from src.models.qsnet.QSNetBreeder import QSNetBreeder
import json


def main():
    with open(opts.breeder_config_path, "r") as fp:
        breeder_config = json.load(fp)

    experiment_factory = ExperimentFactory(model_lookup, loss_function_lookup, optimizer_lookup,
                                           output_transformation_lookup, data_source_delegates_lookup,
                                           trainer_delegates_lookup, evaluation_delegates_lookup,
                                           saver_delegates_lookup)
    breeder = QSNetBreeder(breeder_config, experiment_factory,
                           opts.study_save_path, opts.training_data_path,
                           opts.testing_data_path)
    breeder.start_breeding()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_save_path", default="study_results/study_2")
    parser.add_argument("--breeder_config_path", default="../study_configs/qsnet_breeder.json")
    parser.add_argument("--logging_level", default="INFO")
    opts = parser.parse_args()
    log_level = logging.getLevelName(opts.logging_level)
    logging.basicConfig(level=log_level)
    main()
