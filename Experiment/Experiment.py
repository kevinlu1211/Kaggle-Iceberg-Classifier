from tqdm import tqdm
import itertools

class Experiment(object):

    def __init__(self, model_factory, n_epochs, data_source_delegate, trainer_delegate, evaluation_delegate, saver_delegate,
                 loss_function):
        self.model_factory = model_factory
        self.n_epochs = n_epochs
        self.evaluation_delegate = evaluation_delegate
        self.data_source_delegate = data_source_delegate
        self.saver_delegate = saver_delegate
        self.trainer_delegate = trainer_delegate
        self.loss_function = loss_function
        self.model, self.optimizer, self.scheduler = None, None, None

    def train(self):
        for fold_num, data_fold in enumerate(self.data_source_delegate.retrieve_dataset_for_train()):
            self.model, self.optimizer, self.scheduler = self.model_factory.create_model()
            for epoch in range(self.n_epochs):
                train, val = self.trainer_delegate.on_epoch_start(data_fold)
                for data in tqdm(train):
                    self.model.train()
                    model_output = self.trainer_delegate.create_model_output(data, self.model)
                    transformed_output = self.trainer_delegate.apply_output_transformation(model_output)
                    labels = self.trainer_delegate.create_data_labels(data)
                    model_loss = self.trainer_delegate.calculate_loss(self.loss_function, transformed_output,
                                                                      labels)
                    self.saver_delegate.save_results(data, transformed_output, model_loss, mode="TRAIN")
                    self.trainer_delegate.apply_backward_pass(self.optimizer, model_loss)

                for data in tqdm(val):
                    self.model.eval()
                    model_output = self.trainer_delegate.create_model_output(data, self.model)
                    transformed_output = self.trainer_delegate.apply_output_transformation(model_output)
                    labels = self.trainer_delegate.create_data_labels(data)
                    model_loss = self.trainer_delegate.calculate_loss(self.loss_function, transformed_output,
                                                                      labels)
                    self.saver_delegate.save_results(data, transformed_output, model_loss, mode="VALIDATION")
                self.saver_delegate.on_epoch_end(self.model, epoch, fold_num)
                self.trainer_delegate.update_scheduler(self.saver_delegate.last_epoch_validation_loss, self.scheduler)
                # self.saver_delegate.update_all_results(fold_num, epoch)
                # self.saver_delegate.update_loss_results(fold_num, epoch)
        self.saver_delegate.on_end_experiment()

    def test(self):
        test_data = next(iter(self.data_source_delegate.retrieve_dataset_for_test()))
        self.model, _, _ = self.model_factory.create_model()
        self.model = self.evaluation_delegate.on_setup_model(self.model)
        test = self.evaluation_delegate.on_test_start(test_data)
        for data in tqdm(test):
            self.model.eval()
            model_output = self.trainer_delegate.create_model_output(data, self.model)
            transformed_output = self.trainer_delegate.apply_output_transformation(model_output)
            self.evaluation_delegate.save_evaluation_data(data, transformed_output)
        self.evaluation_delegate.on_test_end()





