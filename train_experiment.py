import argparse
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
        for i, data in enumerate(tqdm(train_loader)):
            (inp, label), _ = data
            models_output, models_losses = experiment_controller.train_in_parallel(inp, label)
            print(models_output)
            print(models_losses)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", default="experiments/experiment_1")
    parser.add_argument("--metadata_folder_path", default="experiment_metadata")
    parser.add_argument("--train_data_fp", default="data/train.json")
    parser.add_argument("--n_epochs", default=10)
    opts = parser.parse_args()
    opts.metadata_file_paths = glob.glob(f"{opts.metadata_folder_path}/*.meta")
    main()
