from abc import ABC, abstractmethod

class AbstractTrainerDelegate(ABC):

    def __init__(self, id, study_save_path):
        self.id = id
        self.experiment_save_path = study_save_path
        self.total_training_loss = []
        self.total_validation_loss = []
        self.training_loss_for_epoch = []
        self.validation_loss_for_epoch = []
        super().__init__()

    @abstractmethod
    def on_experiment_start(self, model):
        pass

    @abstractmethod
    def on_setup_model(self, model):
        pass

    @abstractmethod
    def on_epoch_start(self, dataset):
        pass

    @abstractmethod
    def create_model_input(self, data):
        pass

    @abstractmethod
    def create_data_labels(self, data):
        pass

    @abstractmethod
    def create_model_output(self, model_input, model):
        pass

    @abstractmethod
    def apply_output_transformation(self, model_output, output_transformer):
        pass

    @abstractmethod
    def on_model_output(self, model_output):
        pass

    @abstractmethod
    def calculate_loss(self, loss_function, model_output, labels, mode):
        pass

    @abstractmethod
    def apply_backward_pass(self, optimizer, model_loss, model):
        pass

    @abstractmethod
    def on_epoch_end(self):
        pass

    @abstractmethod
    def on_end_experiment(self):
        pass

