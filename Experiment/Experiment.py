import torch
from torch.autograd import Variable
from pathlib import Path
import os


class Experiment(object):
    def __init__(self, metadata, model, output_transformation, loss_function, optimizer, use_cuda):
        self.metadata = metadata
        if use_cuda:
            self.model = model.cuda()
            self.output_transformation = output_transformation.cuda()
        else:
            self.model = model
            self.output_transformation = output_transformation
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.use_cuda = use_cuda
        self.experiment_outputs = []
        self.experiment_losses = []

    @property
    def name(self):
        return f"{self.metadata['model_name']}-{self.metadata['id']}"

    def train_step(self, inp, label):
        if self.use_cuda:
            inp_data, label = Variable(inp).cuda(), Variable(label).cuda()
        else:
            inp_data, label = Variable(inp), Variable(label)
        model_output = self.model(inp_data)

        # Transform the output here as we will get unnormalized probailities from our models
        # since we don't want our models to be coupled with the data
        transformed_output = self.output_transformation(self.model_output) if self.output_transformation \
                                                                            else model_output
        model_loss = self.loss_function(transformed_output, label)
        self.optimizer.zero_grad()
        model_loss.backward()
        self.optimizer.step()

        return model_output, model_loss

    def eval_step(self, inp, label):
        if self.use_cuda:
            inp_data, label = Variable(inp).cuda(), Variable(label).cuda()
        else:
            inp_data, label = Variable(inp), Variable(label)
        model_output = self.model(inp_data)
        transformed_output = self.output_transformation(self.model_output) if self.output_transformation \
                                                                            else model_output
        return transformed_output

    def save_experiment(self, experiment_path, ckpt_name):

        assert ckpt_name is not None
        model_id = self.metadata["id"]
        models_checkpoint_folder_path = Path(f"{experiment_path}/saved_models/{model_id}")
        models_checkpoint_folder_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), os.fspath(f"{models_checkpoint_folder_path}/{ckpt_name}"))

    def load_experiment(self, experiment_path, ckpt_name, for_eval):

        assert ckpt_name is not None
        model_id = self.metadata["id"]
        model_checkpoint_path = f"{experiment_path}/saved_models/{model_id}/{ckpt_name}"
        self.model.load_state_dict(f"{model_checkpoint_path}")
        if for_eval:
            self.model.eval()




