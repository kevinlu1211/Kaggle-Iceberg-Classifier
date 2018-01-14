## Kaggle Iceberg Classifier 

This repository is a competition hosted on [Kaggle](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge) for 
detecting whether the picture has a iceberg or a ship.

#### Experiment Framework

This repo also includes an Experiment framework which I have written to make my life easier when training
 a lot of different models. The UML diagram of the framework is shown below.
 
 [image1]: ./README_images/Experiment.png 
 [image2]: ./README_images/densenet121_ADAM_dataaug.png
 [image3]: ./README_images/densenet121_ADAM_nodataaug.png
 [image4]: ./README_images/densenet121_ADAM_dataaug_noscheduler.png
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


#### DenseNet121 Experiments

##### Experiment 1

Config file:
```
{
  "model": {
    "name": "DenseNet",
    "model_configuration_name": "DenseNet121",
    "parameters": {
      "block_config": [6, 12, 24, 16],
      "growth_rate": 32,
      "n_init_features": 64,
      "bn_size": 4,
      "dropout_rates": [[0,0,0,0]],
      "n_classes": 1,
      "input_shape": [3, 75, 75]
    }
  },
  "optimizer": {
    "name": "ADAM",
    "parameters": {
      "lr": [0.01, 0.006, 0.003, 0.001, 0.0003, 0.0001, 0.00003],
      "weight_decay": [0.005, 0.0005]
    }
  },
  "scheduler": {
  "name": "ReduceLROnPlateau",
  "parameters": {
    "verbose": true,
    "threshold": 0.03,
    "patience": 7,
    "factor": 0.33
  }
}
  "loss_function": {
    "name": "BCELoss"
  },
  "trainer_delegate": {
    "name": "DenseNet"
  },
  "result_delegate": {
    "name": "DenseNet"
  },
  "data_source_delegate": {
    "name": "DenseNet",
    "parameters": {
      "batch_size": 40,
      "n_splits": 5,
      "splits_to_use": 2,
      "image_size": [75, 75],
      "testing_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/test.json",
      "training_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/train.json"
    }
  },
  "saver_delegate": {
    "name": "DenseNet"
  },
  "n_epochs": 50
}

```
Results:
![alt text][image2]

This was the initial training run and it seems like the best learning rate is around `0.003-0.001` reaching an average loss
of `~0.30`. It seems that the models with a lower learning rate are able to fit the training set better. 
In the next experiment I will use some data augmentation techniques and include dropouts layers in the 
down-sampling/transition layers of DenseNet.

Things to try in the next experiment:
* Try dropout layers after densely connected layers

##### Experiment 2

Config File:

```
{
  "model": {
    "name": "DenseNet",
    "model_configuration_name": "DenseNet121",
    "parameters": {
      "block_config": [6, 12, 24, 16],
      "growth_rate": 32,
      "n_init_features": 64,
      "bn_size": 4,
      "dropout_rates": [[0, 0, 0, 0], [0.05, 0.1, 0.15, 0.20], [0.15, 0.2, 0.25, 0.3],
                        [0.2, 0.15, 0.1, 0.05], [0.3, 0.25, 0.2, 0.15]],
      "n_classes": 1,
      "input_shape": [3, 75, 75]
    }
  },
  "optimizer": {
    "name": "ADAM",
    "parameters": {
      "lr": [0.003, 0.001, 0.0003, 0.0001, 0.00003]
    }
  },
  "scheduler": {
  "name": "ReduceLROnPlateau",
  "parameters": {
    "verbose": true,
    "threshold": 0.03,
    "patience": 7,
    "factor": 0.33
  }
}
  "loss_function": {
    "name": "BCELoss"
  },
  "trainer_delegate": {
    "name": "DenseNet"
  },
  "result_delegate": {
    "name": "DenseNet"
  },
  "data_source_delegate": {
    "name": "DenseNet",
    "parameters": {
      "batch_size": 40,
      "n_splits": 5,
      "splits_to_use": 2,
      "image_size": [75, 75],
      "testing_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/test.json",
      "training_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/train.json"
    }
  },
  "saver_delegate": {
    "name": "DenseNet"
  },
  "n_epochs": 50
}


```

Data Augmentation:

```
transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
```

Results:
![alt text][image3]

So these results weren't that much different to the ones from the previous experiments still having a minimum validation
loss of `~0.3` across the 5 validation folds. One thing I did realise was that the learning rate scheduler seemed a bit 
too aggressive during training as it decreases the learning rate too fast. Also, it seemed like dropout was doing more
harm than good.

In my next experiment:
* Confirm suspicion that dropout isn't helping 
* Don't use a learning rate scheduler
* Instead of doing 5 fold validation, split the training data into 5 even sets, but just pick 2 of them to evaluate on 
as it is taking too long

##### Experiment 3
Config File:
```
{
  "model": {
    "name": "DenseNet",
    "model_configuration_name": "DenseNet121",
    "parameters": {
      "block_config": [6, 12, 24, 16],
      "growth_rate": 32,
      "n_init_features": 64,
      "bn_size": 4,
      "dropout_rates": [[0.30,0.25,0.20,0.15],
                        [0,0,0,0]],
      "n_classes": 1,
      "input_shape": [3, 75, 75]
    }
  },
  "optimizer": {
    "name": "ADAM",
    "parameters": {
      "lr": [0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001],
      "weight_decay": [0]
    }
  },
  "loss_function": {
    "name": "BCELoss"
  },
  "trainer_delegate": {
    "name": "DenseNet"
  },
  "result_delegate": {
    "name": "DenseNet"
  },
  "data_source_delegate": {
    "name": "DenseNet",
    "parameters": {
      "batch_size": 40,
      "n_splits": 5,
      "splits_to_use": 2,
      "image_size": [75, 75],
      "testing_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/test.json",
      "training_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/train.json"
    }
  },
  "saver_delegate": {
    "name": "DenseNet"
  },
  "n_epochs": 50
}

```

Data Augmentation:

```
transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
```
Results:
![alt text][image4]

Not much to say here except I want to explore lower and higher learning rates in the next experiment. Though... it does
seem like with data augmentation we might need a model with more capacity as we can't overfit the training data. Either
that or the data augmentation may be too aggressive, and as a result it is creating too much noise (?). So before I start 
the next experiment I'm going to have a quick look at the transformations. Also I will look at using a less aggressive 
learning rate scheduler

For next experiment:
* Use less aggressive learning rate scheduler
* Try model with higher capacity
