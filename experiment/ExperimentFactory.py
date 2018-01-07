from uuid import uuid4
from .Experiment import Experiment
from src.utils import cudarize


class ExperimentFactory(object):

    with_parameters = lambda a, b: a(**b) if b else a()
    create_component = lambda x, y: ExperimentFactory.with_parameters(x, y) if x else None
    """
    The above is equivalent to:
    def with_parameters(component_ptr, component_parameters):
        if component_parameters is None:
            return component_ptr()
        else:
            return component_ptr(**component_parameters) 
            
    def create_component(component_ptr, component_parameters):
        if component is None:
            return None
        else:
            return with_parameters(component_ptr, component_parameters)
    """

    def __init__(self, model_lookup, loss_function_lookup,
                 optimizer_lookup, scheduler_lookup, data_source_delegate_lookup, trainer_delegate_lookup,
                 evaluation_delegate_lookup, saver_delegates_lookup):

        self.model_lookup = model_lookup
        self.loss_function_lookup = loss_function_lookup
        self.optimizer_lookup = optimizer_lookup
        self.scheduler_lookup = scheduler_lookup
        self.data_source_delegate_lookup = data_source_delegate_lookup
        self.trainer_delegate_lookup = trainer_delegate_lookup
        self.evaluation_delegate_lookup = evaluation_delegate_lookup
        self.saver_delegate_lookup = saver_delegates_lookup

    def create_experiment(self, experiment_config, study_save_path,
                          model_load_path=None, eval_save_path=None):

        # Get the experiment id
        experiment_id = experiment_config.get('experiment_id', str(uuid4()))

        # Create the loss function
        loss_function_name = experiment_config.get("loss_function").get("name")
        loss_function_parameters = experiment_config.get("loss_function").get("parameters")
        loss_function_ptr = self.loss_function_lookup.get(loss_function_name)
        loss_function = ExperimentFactory.create_component(loss_function_ptr, loss_function_parameters)

        # Create the data delegate
        data_source_delegate_name = experiment_config.get("data_source_delegate").get("name")
        data_source_delegate_parameters = experiment_config.get("data_source_delegate").get("parameters", {})
        data_source_delegate_ptr = self.data_source_delegate_lookup.get(data_source_delegate_name)
        data_source_delegate = ExperimentFactory.create_component(data_source_delegate_ptr,
                                                                  data_source_delegate_parameters)

        # Create the trainer delegate
        trainer_delegate_name = experiment_config.get("trainer_delegate").get("name")
        trainer_delegate_parameters = experiment_config.get("trainer_delegate").get("parameters", {})
        trainer_delegate_parameters["experiment_id"] = experiment_id
        trainer_delegate_parameters["study_save_path"] = study_save_path
        trainer_delegate_ptr = self.trainer_delegate_lookup.get(trainer_delegate_name)
        trainer_delegate = ExperimentFactory.create_component(trainer_delegate_ptr,
                                                              trainer_delegate_parameters)

        # Create the savers delegate
        saver_delegate_name = experiment_config.get("saver_delegate", dict()).get("name")
        saver_delegate_parameters = experiment_config.get("saver_delegate", dict()).get("parameters", dict())
        saver_delegate_parameters["experiment_id"] = experiment_id
        saver_delegate_parameters["study_save_path"] = study_save_path
        saver_delegate_parameters["experiment_config"] = experiment_config
        saver_delegate_ptr = self.saver_delegate_lookup.get(saver_delegate_name)
        saver_delegate = ExperimentFactory.create_component(saver_delegate_ptr,
                                                            saver_delegate_parameters)
        # Create the evaluator delegate
        evaluation_delegate = experiment_config.get("evaluation_delegate")
        if evaluation_delegate is not None:
            evaluation_delegate_name = evaluation_delegate.get("name")
            evaluation_delegate_parameters = evaluation_delegate.get("parameters", {})
            evaluation_delegate_parameters["model_load_path"] = model_load_path
            evaluation_delegate_parameters["eval_save_path"] = eval_save_path
            evaluation_delegate_ptr = self.evaluation_delegate_lookup.get(evaluation_delegate_name)
            evaluation_delegate = ExperimentFactory.create_component(evaluation_delegate_ptr,
                                                                     evaluation_delegate_parameters)

        model_factory = ModelFactory(experiment_config, self.model_lookup, self.optimizer_lookup, self.scheduler_lookup)
        return Experiment(model_factory, experiment_config.get("n_epochs"), data_source_delegate,
                          trainer_delegate, evaluation_delegate, saver_delegate,
                          loss_function)

class ModelFactory(object):
    def __init__(self, experiment_config, model_lookup, optimizer_lookup, scheduler_lookup):
        self.experiment_config = experiment_config
        self.model_lookup = model_lookup
        self.optimizer_lookup = optimizer_lookup
        self.scheduler_lookup = scheduler_lookup

    def create_model(self):
        # Create the model
        model_name = self.experiment_config.get("model").get("name")
        model_config = self.experiment_config.get("model").get("parameters")
        model_ptr = self.model_lookup[model_name]
        model = ExperimentFactory.create_component(model_ptr, model_config)

        # Create the optimizer
        optimizer_name = self.experiment_config.get("optimizer").get("name")
        optimizer_parameters = self.experiment_config.get("optimizer").get("parameters", {})
        optimizer_parameters["params"] = model.parameters()
        optimizer_ptr = self.optimizer_lookup.get(optimizer_name)
        optimizer = ExperimentFactory.create_component(optimizer_ptr, optimizer_parameters)

        # Create Scheduler
        scheduler_name = self.experiment_config.get("scheduler", dict()).get("name")
        if scheduler_name is not None:
            scheduler_parameters = self.experiment_config.get("scheduler", dict()).get("parameters")
            scheduler_parameters['optimizer'] = optimizer
            scheduler_ptr = self.scheduler_lookup.get(scheduler_name)
            scheduler = ExperimentFactory.create_component(scheduler_ptr, scheduler_parameters)
        else:
            scheduler = None
        model = cudarize(model)
        return model, optimizer, scheduler
