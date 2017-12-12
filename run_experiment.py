import argparse
import logging
import time
from src.experiment_mappings \
    import model_lookup, \
    loss_function_lookup, \
    optimizer_lookup, \
    output_transformation_lookup, \
    data_source_delegates_lookup, \
    trainer_delegates_lookup, \
    evaluation_delegates_lookup

from src.Experiment import ExperimentFactory

def main():
    experiment_factory = ExperimentFactory(model_lookup, loss_function_lookup, optimizer_lookup,
                                           output_transformation_lookup, data_source_delegates_lookup,
                                           trainer_delegates_lookup, evaluation_delegates_lookup)
    experiment = experiment_factory.create_experiment(opts.config_file_path, opts.experiment_path,
                                                      opts.training_data_path, opts.testing_data_path)
    experiment.train()
    experiment.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", default="../experiments_output/experiment_1")
    parser.add_argument("--config_file_path", default="../experiment_config/experiment_1/densenet.json")
    parser.add_argument("--training_data_path", default="data/train.json")
    parser.add_argument("--testing_data_path", default=None)
    parser.add_argument("--logging_level", default="INFO")
    opts = parser.parse_args()
    opts.experiment_path = f"{opts.experiment_path}/{time.strftime('%Y%m%d-%H%M%S')}"

    # opts.metadata_file_paths = glob.glob(f"{opts.metadata_folder_path}/*.json")
    log_level = logging.getLevelName(opts.logging_level)
    logging.basicConfig(level=log_level)
    main()
