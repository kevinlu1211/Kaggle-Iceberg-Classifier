from tqdm import tqdm
import itertools
from copy import deepcopy

class Experiment(object):

    def __init__(self, n_epochs, data_source_delegate, trainer_delegate, evaluation_delegate, saver_delegate,
                 model, output_transformation, loss_function, optimizer, scheduler):
        self.n_epochs = n_epochs
        self.evaluation_delegate = evaluation_delegate
        self.data_source_delegate = data_source_delegate
        self.saver_delegate = saver_delegate
        self.trainer_delegate = trainer_delegate
        self.model = model
        self.output_transformation = output_transformation
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self):
        self.trainer_delegate.on_experiment_start(self.model)
        for fold_num, data_fold in enumerate(self.data_source_delegate.retrieve_dataset_for_train()):
            self.model = self.trainer_delegate.on_setup_model(self.model)
            # self.optimizer = deepcopy(self._optimizer)
            # self.scheduler = deepcopy(self._scheduler)
            for epoch in range(self.n_epochs):
                train, val = self.trainer_delegate.on_epoch_start(data_fold)
                for data in tqdm(train):
                    self.model.train()
                    model_input = self.trainer_delegate.create_model_input(data)
                    model_output = self.trainer_delegate.create_model_output(model_input, self.model)
                    transformed_output = self.trainer_delegate.apply_output_transformation(model_output,
                                                                                           self.output_transformation)
                    self.trainer_delegate.on_model_output(transformed_output)
                    labels = self.trainer_delegate.create_data_labels(data)
                    training_loss = self.trainer_delegate.calculate_loss(self.loss_function, transformed_output,
                                                                      labels, mode="TRAIN")
                    self.saver_delegate.on_model_loss(training_loss, mode="TRAIN")
                    self.trainer_delegate.apply_backward_pass(self.optimizer, self.scheduler, training_loss, self.model)

                for data in tqdm(val):
                    self.model.eval()
                    model_input = self.trainer_delegate.create_model_input(data)
                    model_output = self.trainer_delegate.create_model_output(model_input, self.model)
                    transformed_output = self.trainer_delegate.apply_output_transformation(model_output,
                                                                                           self.output_transformation)
                    self.trainer_delegate.on_model_output(transformed_output)
                    labels = self.trainer_delegate.create_data_labels(data)
                    validation_loss = self.trainer_delegate.calculate_loss(self.loss_function, transformed_output,
                                                                           labels, mode="EVAL")
                    self.saver_delegate.on_model_loss(validation_loss, mode="EVAL")
                self.saver_delegate.on_epoch_end(self.model, epoch, fold_num)
                self.trainer_delegate.update_scheduler(self.saver_delegate.last_epoch_validation_loss,
                                                       self.scheduler)
        self.trainer_delegate.on_end_experiment()

    def test(self):
        test_data = list(self.data_source_delegate.retrieve_dataset_for_test())[0]
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





