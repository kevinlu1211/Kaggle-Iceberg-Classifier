import argparse
import logging
import glob as glob
from src.experiment_mappings \
    import model_lookup, \
    loss_function_lookup, \
    optimizer_lookup, \
    output_transformation_lookup, \
    data_source_delegates_lookup, \
    trainer_delegates_lookup

from src.experiment import ExperimentFactory

import torch
use_cuda = torch.cuda.is_available()


def main():
    experiment_factory = ExperimentFactory(model_lookup, loss_function_lookup, optimizer_lookup,
                                           output_transformation_lookup, data_source_delegates_lookup,
                                           trainer_delegates_lookup, use_cuda=use_cuda)
    experiment = experiment_factory.create_experiment(opts.config_file_path, opts.experiment_path, opts.training_data_path)
    experiment.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", default="../experiments_output/experiment_1")
    parser.add_argument("--config_file_path", default="../experiment_config/experiment_1/experiment_1_qsnet.json")
    parser.add_argument("--training_data_path", default="data/train.json")
    parser.add_argument("--logging_level", default="INFO")
    opts = parser.parse_args()
    # opts.metadata_file_paths = glob.glob(f"{opts.metadata_folder_path}/*.json")
    log_level = logging.getLevelName(opts.logging_level)
    logging.basicConfig(level=log_level)
    main()
