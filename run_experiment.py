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
    result_delegates_lookup
from src.Experiment import ExperimentFactory
from src.models.qsnet.Breeder import QSNetBreeder
import json


def main():
    with open(opts.config_file_path, "r") as fp:
        breeder_config = json.load(fp)

    experiment_factory = ExperimentFactory(model_lookup, loss_function_lookup, optimizer_lookup,
                                           output_transformation_lookup, data_source_delegates_lookup,
                                           trainer_delegates_lookup, evaluation_delegates_lookup,
                                           result_delegates_lookup)
    breeder = QSNetBreeder(breeder_config, experiment_factory,
                           opts.study_save_path, opts.training_data_path,
                           opts.testing_data_path)
    breeder.start_breeding()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_save_path", default="../study_results/study_1")
    parser.add_argument("--config_file_path", default="../study_configs/study_1/qsnet_breeder.json")
    parser.add_argument("--training_data_path", default="data/train.json")
    parser.add_argument("--testing_data_path", default=None)
    parser.add_argument("--logging_level", default="INFO")
    opts = parser.parse_args()
    opts.experiment_path = f"{opts.study_save_path}/{time.strftime('%Y%m%d-%H%M%S')}"

    # opts.metadata_file_paths = glob.glob(f"{opts.metadata_folder_path}/*.json")
    log_level = logging.getLevelName(opts.logging_level)
    logging.basicConfig(level=log_level)
    main()
