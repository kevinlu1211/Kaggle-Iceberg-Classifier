import argparse
from collections import defaultdict
import logging
import glob as glob
from experiment_mappings \
    import model_lookup, \
    model_preprocessing_lookup, \
    loss_function_lookup, \
    optimizer_lookup, \
    output_transformation_lookup

from experiment import ExperimentsController
from experiment import ExperimentFactory
from utils.dataset import create_train_val_dataloaders
from tqdm import tqdm
import torch
use_cuda = torch.cuda.is_available()

def main():
    experiment_factory = ExperimentFactory(model_lookup, model_preprocessing_lookup,
                                       loss_function_lookup, optimizer_lookup,
                                       output_transformation_lookup, use_cuda=use_cuda)
    experiment_controller = ExperimentsController(experiment_factory)
    experiment_controller.setup_experiments(opts.experiment_path, opts.metadata_file_paths)
    train_loader, val_loader = create_train_val_dataloaders(opts.train_data_fp, batch_size=32)


    for epoch in range(opts.n_epochs):
        models_training_loss_for_epoch = defaultdict(list)
        models_val_loss_for_epoch = defaultdict(list)
        for i, data in enumerate(tqdm(train_loader)):
            (inp, label), _ = data
            models_output, models_losses = experiment_controller.train_in_parallel(inp, label)
            for model_name, model_loss in models_losses.items():
                models_training_loss_for_epoch[model_name].append(model_loss.data[0])

        for model_name, model_loss_for_epoch in models_training_loss_for_epoch.items():
            avg_loss_for_model = sum(model_loss_for_epoch)/len(model_loss_for_epoch)
            logging.info(f"Average loss in Epoch {epoch} for {model_name} is {avg_loss_for_model:.2f}")

        for i, data in enumerate(tqdm(val_loader)):
            (inp, label), _ = data
            models_output, models_losses = experiment_controller.val_in_parallel(inp, label)
            for model_name, model_loss in models_losses.items():
                models_val_loss_for_epoch[model_name].append(model_loss.data[0])

        for model_name, model_loss_for_epoch in models_val_loss_for_epoch.items():
            avg_loss_for_model = sum(model_loss_for_epoch) / len(model_loss_for_epoch)
            logging.info(f"Eval loss for Epoch {epoch} for {model_name} is {avg_loss_for_model:.2f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", default="../experiments/experiment_1")
    parser.add_argument("--metadata_folder_path", default="../experiment_metadata/experiment_1")
    parser.add_argument("--train_data_fp", default="data/train.json")
    parser.add_argument("--n_epochs", default=10)
    parser.add_argument("--logging_level", default="INFO")
    opts = parser.parse_args()
    opts.metadata_file_paths = glob.glob(f"{opts.metadata_folder_path}/*.meta")
    log_level = logging.getLevelName(opts.logging_level)
    logging.basicConfig(level=log_level)
    main()
