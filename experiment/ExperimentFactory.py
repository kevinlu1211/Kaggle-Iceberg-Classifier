from pathlib import Path
import json
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
                 optimizer_lookup, output_transformation_lookup,
                 data_source_delegate_lookup, trainer_delegate_lookup,
                 evaluation_delegate_lookup, saver_delegates_lookup):

        self.model_lookup = model_lookup
        self.loss_function_lookup = loss_function_lookup
        self.optimizer_lookup = optimizer_lookup
        self.output_transformation_lookup = output_transformation_lookup
        self.data_source_delegate_lookup = data_source_delegate_lookup
        self.trainer_delegate_lookup = trainer_delegate_lookup
        self.evaluation_delegate_lookup = evaluation_delegate_lookup
        self.saver_delegate_lookup = saver_delegates_lookup

    def create_experiment(self, experiment_config, study_save_path,
                          model_load_path=None, eval_save_path=None):

        # Create the model
        model_name = experiment_config.get("model").get("name")
        model_config = experiment_config.get("model").get("config")
        model_ptr = self.model_lookup[model_name]
        model = model_ptr(model_config)

        # Create the optimizer
        optimizer_name = experiment_config.get("optimizer").get("name")
        optimizer_parameters = experiment_config.get("optimizer").get("parameters", {})
        optimizer_parameters["params"] = model.parameters()
        optimizer_ptr = self.optimizer_lookup.get(optimizer_name)
        optimizer = ExperimentFactory.create_component(optimizer_ptr, optimizer_parameters)

        # Create the loss function
        loss_function_name = experiment_config.get("loss_function").get("name")
        loss_function_parameters = experiment_config.get("loss_function").get("parameters")
        loss_function_ptr = self.loss_function_lookup.get(loss_function_name)
        loss_function = ExperimentFactory.create_component(loss_function_ptr, loss_function_parameters)

        # Create the output transformation function
        output_transformation_name = experiment_config.get("output_transformation").get("name")
        output_transformation_parameters = experiment_config.get("output_transformation").get("parameters")
        output_transformation_ptr = self.output_transformation_lookup.get(output_transformation_name)
        output_transformation = ExperimentFactory.create_component(output_transformation_ptr,
                                                                   output_transformation_parameters)

        # Create the data delegate
        data_source_delegate_name = experiment_config.get("data_source_delegate").get("name")
        data_source_delegate_parameters = experiment_config.get("data_source_delegate").get("parameters", {})
        data_source_delegate_ptr = self.data_source_delegate_lookup.get(data_source_delegate_name)
        data_source_delegate = ExperimentFactory.create_component(data_source_delegate_ptr,
                                                                  data_source_delegate_parameters)

        # Create the trainer delegate
        trainer_delegate_name = experiment_config.get("trainer_delegate").get("name")
        trainer_delegate_parameters = experiment_config.get("trainer_delegate").get("parameters", {})
        trainer_delegate_parameters["experiment_id"] = experiment_config["id"]
        trainer_delegate_parameters["study_save_path"] = study_save_path
        trainer_delegate_ptr = self.trainer_delegate_lookup.get(trainer_delegate_name)
        trainer_delegate = ExperimentFactory.create_component(trainer_delegate_ptr,
                                                              trainer_delegate_parameters)

        # Create the savers delegate
        saver_delegate_name = experiment_config.get("saver_delegate", dict()).get("name")
        saver_delegate_parameters = experiment_config.get("saver_delegate", dict()).get("parameters", dict())
        saver_delegate_parameters["experiment_id"] = experiment_config["id"]
        saver_delegate_parameters["study_save_path"] = study_save_path
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

        model = cudarize(model)
        output_transformation = cudarize(output_transformation)

        return Experiment(experiment_config.get("n_epochs"), data_source_delegate,
                          trainer_delegate, evaluation_delegate, saver_delegate,
                          model, output_transformation, loss_function, optimizer)

