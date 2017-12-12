from tqdm import tqdm
from copy import deepcopy

class Experiment(object):
    def __init__(self, n_epochs, data_source_delegate, trainer_delegate, evaluation_delegate,
                 model, output_transformation, loss_function, optimizer):
        self.n_epochs = n_epochs
        self.evaluation_delegate = evaluation_delegate
        self.data_source_delegate = data_source_delegate
        self.trainer_delegate = trainer_delegate
        self.model = model
        self.output_transformation = output_transformation
        self.loss_function = loss_function
        self.optimizer = optimizer


    def train(self):
        self.trainer_delegate.on_experiment_start(self.model)
        for data_split in self.data_source_delegate.retrieve_dataset_for_train():
            self.model = self.trainer_delegate.on_setup_model(self.model)
            for epoch in range(self.n_epochs):
                train, val = self.trainer_delegate.on_epoch_start(data_split)
                for data in tqdm(train):
                    self.model.train()
                    model_input = self.trainer_delegate.create_model_input(data)
                    model_output = self.trainer_delegate.create_model_output(model_input, self.model)
                    transformed_output = self.trainer_delegate.apply_output_transformation(model_output,
                                                                                           self.output_transformation)
                    self.trainer_delegate.on_model_output(transformed_output)
                    labels = self.trainer_delegate.create_data_labels(data)
                    model_loss = self.trainer_delegate.calculate_loss(self.loss_function, transformed_output,
                                                                      labels, mode="TRAIN")
                    self.trainer_delegate.apply_backward_pass(self.optimizer, model_loss, self.model)

                for data in tqdm(val):
                    self.model.eval()
                    model_input = self.trainer_delegate.create_model_input(data)
                    model_output = self.trainer_delegate.create_model_output(model_input, self.model)
                    transformed_output = self.trainer_delegate.apply_output_transformation(model_output,
                                                                                           self.output_transformation)
                    self.trainer_delegate.on_model_output(transformed_output)
                    labels = self.trainer_delegate.create_data_labels(data)
                    _ = self.trainer_delegate.calculate_loss(self.loss_function, transformed_output,
                                                             labels, mode="EVAL")

                self.trainer_delegate.on_epoch_end(self.model, epoch)
        self.trainer_delegate.on_end_experiment()

    def test(self):
        test_data = self.data_source_delegate.retrieve_dataset_for_test()
        self.model = self.evaluation_delegate.on_setup_model(self.model)
        test = self.evaluation_delegate.on_test_start(test_data)
        for data in tqdm(test):
            self.model.eval()
            model_input = self.trainer_delegate.create_model_input(data)
            model_output = self.trainer_delegate.create_model_output(model_input, self.model)
            transformed_output = self.trainer_delegate.apply_output_transformation(model_output,
                                                                                   self.output_transformation)
            self.evaluation_delegate.save_evaluation_data(data, transformed_output)
        self.evaluation_delegate.on_test_end()





