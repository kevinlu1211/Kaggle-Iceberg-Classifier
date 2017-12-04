from copy import deepcopy
import json
from .Experiment import Experiment


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
                 optimizer_lookup, output_transformation_lookup, data_source_delegate_lookup,
                 trainer_delegate_lookup, use_cuda):

        self.model_lookup = model_lookup
        self.loss_function_lookup = loss_function_lookup
        self.optimizer_lookup = optimizer_lookup
        self.output_transformation_lookup = output_transformation_lookup
        self.data_source_delegate_lookup = data_source_delegate_lookup
        self.trainer_delegate_lookup = trainer_delegate_lookup
        self.use_cuda = use_cuda

    def create_experiment(self, experiment_config_path, experiment_save_path, training_data_path):

        # Read config file
        with open(experiment_config_path, "r") as fp:
            experiment_config = json.load(fp)

        # Make a shallow copy so that we don't change the original experiment_config, since we can't json.dump
        # the "params" key value pair for the optimizer
        experiment_config = deepcopy(experiment_config)

        # Create the model
        model_name = experiment_config.get("model_name")
        model_parameters = experiment_config.get("model_parameters")
        model_ptr = self.model_lookup[model_name]
        model = ExperimentFactory.create_component(model_ptr, model_parameters)

        # Create the optimizer
        optimizer_name = experiment_config.get("optimizer_name")
        optimizer_parameters = experiment_config.get("optimizer_parameters", {})
        optimizer_parameters["params"] = model.parameters()
        optimizer_ptr = self.optimizer_lookup.get(optimizer_name)
        optimizer = ExperimentFactory.create_component(optimizer_ptr, optimizer_parameters)

        # Create the loss function
        loss_function_name = experiment_config.get("loss_function_name")
        loss_function_parameters = experiment_config.get("loss_function_parameters")
        loss_function_ptr = self.loss_function_lookup.get(loss_function_name)
        loss_function = ExperimentFactory.create_component(loss_function_ptr, loss_function_parameters)

        # Create the output transformation function
        output_transformation_name = experiment_config.get("output_transformation_name")
        output_transformation_parameters = experiment_config.get("output_transformation_parameters")
        output_transformation_ptr = self.output_transformation_lookup.get(output_transformation_name)
        output_transformation = ExperimentFactory.create_component(output_transformation_ptr,
                                                                   output_transformation_parameters)

        # Create the data delegate
        data_source_delegate_name = experiment_config.get("data_source_delegate_name")
        data_source_delegate_parameters = experiment_config.get("data_source_delegate_parameters", {})
        data_source_delegate_parameters["training_data_path"] = training_data_path
        data_source_delegate_ptr = self.data_source_delegate_lookup.get(data_source_delegate_name)
        data_source_delegate = ExperimentFactory.create_component(data_source_delegate_ptr,
                                                                  data_source_delegate_parameters)

        # Create the trainer delegate
        trainer_delegate_name = experiment_config.get("trainer_delegate_name")
        trainer_delegate_parameters = experiment_config.get("trainer_delegate_parameters", {})
        trainer_delegate_parameters["experiment_save_path"] = experiment_save_path
        trainer_delegate_ptr = self.trainer_delegate_lookup.get(trainer_delegate_name)
        trainer_delegate = ExperimentFactory.create_component(trainer_delegate_ptr,
                                                           trainer_delegate_parameters)
        if self.use_cuda:
            self.model = model.cuda()
            self.output_transformation = output_transformation.cuda()
        else:
            self.model = model
            self.output_transformation = output_transformation

        return Experiment(experiment_config.get("n_epochs"), data_source_delegate,
                          trainer_delegate, model, output_transformation, loss_function, optimizer)

