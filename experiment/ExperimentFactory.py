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

    def __init__(self, model_ptrs, model_preprocessing_ptrs, loss_function_ptrs,
                 optimizer_ptrs, output_transformation_lookup, use_cuda):
        self.model_ptrs = model_ptrs
        self.model_preprocessing_ptrs = model_preprocessing_ptrs
        self.loss_function_ptrs = loss_function_ptrs
        self.optimizer_ptrs = optimizer_ptrs
        self.output_transformation_lookup = output_transformation_lookup
        self.use_cuda = use_cuda


    def create_experiment(self, experiment_metadata):

        # Create the model
        model_name = experiment_metadata.get("model_name")
        model_parameters = experiment_metadata.get("model_parameters")
        model_ptr = self.model_ptrs[model_name]
        model = ExperimentFactory.create_component(model_ptr, model_parameters)

        # Create the optimizer
        optimizer_name = experiment_metadata.get("optimizer_name")
        optimizer_parameters = experiment_metadata.get("optimizer_parameters", {})
        optimizer_parameters["params"] = model.parameters()
        optimizer_ptr = self.optimizer_ptrs.get(optimizer_name)
        optimizer = ExperimentFactory.create_component(optimizer_ptr, optimizer_parameters)

        # Create the loss function
        loss_function_name = experiment_metadata.get("loss_function_name")
        loss_function_parameters = experiment_metadata.get("loss_function_parameters")
        loss_function_ptr = self.loss_function_ptrs.get(loss_function_name)
        loss_function = ExperimentFactory.create_component(loss_function_ptr, loss_function_parameters)

        # Create the output transformation function
        output_transformation_name = experiment_metadata.get("output_transformation_name")
        output_transformation_parameters = experiment_metadata.get("output_transformation_parameters")
        output_transformation_ptr = self.output_transformation_lookup.get(output_transformation_name)
        output_transformation = ExperimentFactory.create_component(output_transformation_ptr,
                                                                   output_transformation_parameters)

        return Experiment(experiment_metadata, model, output_transformation,
                          optimizer, loss_function, use_cuda=self.use_cuda)

