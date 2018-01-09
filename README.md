## Kaggle Iceberg Classifier 

This repository is a competition hosted on [Kaggle](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge) for 
detecting whether the picture has a iceberg or a ship.

#### Experiment Framework

This repo also includes an Experiment framework which I have written to make my life easier when training
 a lot of different models. The UML diagram of the framework is shown below.
 
 [image1]: ./README_images/Experiment.png 
 [image2]: ./README_images/Densenet121.png
 ![alt text][image1]


The main purpose of the framework is to create an end-to-end pipeline (data preprocessing -> model training -> 
evaluating model performance) that can easily manipulated. As a result of the lookup dictionaries being injected
into the ExperimentFactory I can mix and match different kinds of preprocessing methods, optimizer, schedulers, and 
model architectures without having to write extra boilerplate code. Moreover, due to the Experiment Configuration File
I will easily be able to re-run past experiments, just by passing the configuration file into the ExperimentFactory


#### How the Configuration File works
Examples of configuration file can be found in the `study_config` folder. This file works by providing the information 
needed by `Experiment/ExperimentFactory.py` to create an Experiment. 

For example in `study_configs/densenet121_experiment.json` there is an entry for the optimizer.
```
"optimizer": {
    "name": "SGD",
    "parameters": {
        "lr": 0.01
      }
  }
```

The optimizer object is created in the `ExperimentFactory` by:
```
with_parameters = lambda a, b: a(**b) if b else a()
create_component = lambda x, y: ExperimentFactory.with_parameters(x, y) if x else None
optimizer_name = self.experiment_config.get("optimizer").get("name")
optimizer_parameters = self.experiment_config.get("optimizer").get("parameters", {})
optimizer_parameters["params"] = model.parameters()
optimizer_ptr = self.optimizer_lookup.get(optimizer_name)
optimizer = ExperimentFactory.create_component(optimizer_ptr, optimizer_parameters)
```

Where `self.optimizer_lookup` is just the dictionary named `optimizers` in `ExperimentMappings/optimizers.py`

```
optimizers = {
    "ADAM": torch.optim.Adam,
    "RMSprop": torch.optim.RMSprop,
    "SGD": torch.optim.SGD
}
```


#### DenseNet121 results for different hyperparameters
![alt text][image2]


