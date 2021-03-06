## Kaggle Iceberg Classifier 

This repository is a competition hosted on [Kaggle](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge) for 
detecting whether the picture has a iceberg or a ship.

#### Final rank: 435/3342 

#### Things to try:
* ~~Try and optimize over different kinds of neural network architectures~~
* ~~Try XGBoost and SVMs (with HOG/SIFT features) and compare results to CNN~~
* ~~Explore stacking with different classifier (hopefully XGBoost and SVMs will have low correlation with the CNNs)~~
* Try to do psuedo-labelling on the test set to create more training data
* Try expanding number of samples by implementing [Data augmentation by pairing samples
for images classification](https://arxiv.org/pdf/1801.02929.pdf)

#### Experiment Framework

This repo also includes an Experiment framework which I have written to make my life easier when training
 a lot of different models. The UML diagram of the framework is shown below.
 
 [image1]: ./README_images/Experiment.png 
 [image2]: ./README_images/densenet121_ADAM_dataaug.png
 [image3]: ./README_images/densenet121_ADAM_nodataaug.png
 [image4]: ./README_images/densenet121_ADAM_dataaug_noscheduler.png
 ![alt text][image1]


The main purpose of the framework is to create an end-to-end pipeline (data pre-processing -> model training -> 
evaluating model performance) that can easily manipulated. As a result of the lookup dictionaries being injected
into the ExperimentFactory I can mix and match different kinds of pre-processing methods, optimizer, schedulers, and 
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
    "name": "StatOil"
  },
  "result_delegate": {
    "name": "StatOil"
  },
  "data_source_delegate": {
    "name": "StatOil",
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
    "name": "StatOil"
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
    "name": "StatOil"
  },
  "result_delegate": {
    "name": "StatOil"
  },
  "data_source_delegate": {
    "name": "StatOil",
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
    "name": "StatOil"
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
    "name": "StatOil"
  },
  "result_delegate": {
    "name": "StatOil"
  },
  "data_source_delegate": {
    "name": "StatOil",
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
    "name": "StatOil"
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

It seems like dropout doesn't really help as seen by the variance in the validation performance. Though the main thing
to point out is that no matter what learning rate is used the validation error doesn't seem to be able to decrease below
`~0.3`. For my next step I'm going to run a series of experiments similar to what I have done above, but with 
[Squeeze-Excitation Networks](https://arxiv.org/abs/1709.01507), and also use a less aggressive learning rate scheduler

For next experiment:
* Use less aggressive learning rate scheduler
* Try model with higher capacity

##### Experiment 4

Experiment Config:
```
{
  "model": {
    "name": "IceResNet",
    "model_configuration_name": "IceResNet",
    "parameters": {
      "num_classes": 1,
      "num_rgb": 3,
      "base": 32
    }
  },
  "optimizer": {
    "name": "ADAM",
    "parameters": {
      "lr": [0.005, 0.0005],
      "weight_decay": [0.00005]
    }
  },
  'scheduler': {'name': 'ReduceLROnPlateau',
      'parameters': {'factor': 0.33,
       'patience': 15,
       'threshold': 0.1,
       'verbose': True
       }
  },
  "loss_function": {
    "name": "BCELoss"
  },
  "trainer_delegate": {
    "name": "StatOil"
  },
  "result_delegate": {
    "name": "StatOil"
  },
  "data_source_delegate": {
    "name": "StatOil",
    "parameters": {
      "batch_size": 40,
      "n_splits": 5,
      "splits_to_use": 5,
      "image_size": [75, 75],
      "testing_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/test.json",
      "training_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/train.json"
    }
  },
  "saver_delegate": {
    "name": "StatOil"
  },
  "n_epochs": 80
}
```

Data Augmentation:

```
transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
```
[image5]: ./README_images/iceresnet_ADAM_no_scheduler_no_dataaug.png
[image6]: ./README_images/iceresnet_ADAM_no_scheduler_dataaug.png
[image7]: ./README_images/iceresnet_ADAM_scheduler_dataaug.png
[image8]: ./README_images/iceresnet_ADAM_no_scheduler_dataaug_no_rotation.png

Results:

No learning rate scheduler and no data augmentation
![alt_text][image5]

No learning rate scheduler and data augmentation
![alt_text][image6]

Learning rate scheduler and data augmentation
![alt_text][image7]

Learning rate scheduler and data augmentation (no rotation)
![alt_text][image8]

Again it really seems like we are just getting stuck at a loss of `~0.3`. One thing that I haven't messed around with is
changing the data augmentation step. As the `ToTensor()` method of PyTorch rescales the images to values `[0, 1]` 
and the values of the image aren't this may cause some loss in the information.

For next experiment:
* Don't use PyTorch's in-built transform functions

##### Experiment 5

Experiment Config:
```angular2html
{
  "model": {
    "name": "IceResNet",
    "model_configuration_name": "IceResNet",
    "parameters": {
      "num_classes": 1,
      "num_rgb": 3,
      "base": 32
    }
  },
  "optimizer": {
    "name": "ADAM",
    "parameters": {
      "lr": [0.0005],
      "weight_decay": [0.00005]
    }
  },
  "scheduler": {"name": "ReduceLROnPlateau",
      "parameters": {"factor": 0.5,
       "patience": 15,
       "threshold": 0.1,
       "verbose": true
       }
  },
  "loss_function": {
    "name": "BCELoss"
  },
  "trainer_delegate": {
    "name": "StatOil"
  },
  "result_delegate": {
    "name": "StatOil"
  },
  "data_source_delegate": {
    "name": "StatOil",
    "parameters": {
      "batch_size": 40,
      "n_splits": 5,
      "splits_to_use": 5,
      "image_size": [75, 75],
      "data_handler_method": "ThreeChannels",
      "testing_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/test.json",
      "training_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/train.json"
    }
  },
  "saver_delegate": {
    "name": "StatOil"
  },
  "n_epochs": 80
}
```

Data Augmentation:
```angular2html
 transforms.Compose([
            horizontal_flip,
            convert_image_to_tensor
        ])
```

Results:

[image9]: ./README_images/iceresnet_ADAM_scheduler_newdataaug.png
![alt_text][image9]

Wow definitely a big change in our loss, reaching `~0.2`. The next thing to try would be to see if only using two 
channels would be better.

For next experiment:
* Only use two channels with a variety of learning rates

##### Experiment 6

Experiment Config:
```angular2html
{
  "model": {
    "name": "IceResNet",
    "model_configuration_name": "IceResNet",
    "parameters": {
      "num_classes": 1,
      "num_rgb": 2,
      "base": 32
    }
  },
  "optimizer": {
    "name": "ADAM",
    "parameters": {
      "lr": [0.005, 0.0005, 0.00005],
      "weight_decay": [0.00005]
    }
  },
  "scheduler": {"name": "ReduceLROnPlateau",
      "parameters": {"factor": 0.5,
       "patience": 15,
       "threshold": 0.1,
       "verbose": true
       }
  },
  "loss_function": {
    "name": "BCELoss"
  },
  "trainer_delegate": {
    "name": "StatOil"
  },
  "result_delegate": {
    "name": "StatOil"
  },
  "data_source_delegate": {
    "name": "StatOil",
    "parameters": {
      "batch_size": 40,
      "n_splits": 5,
      "splits_to_use": 5,
      "image_size": [75, 75],
      "data_handler_method": "TwoChannels",
      "testing_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/test.json",
      "training_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/train.json"
    }
  },
  "saver_delegate": {
    "name": "StatOil"
  },
  "n_epochs": 80
}
```

Data Augmentation:
```angular2html
 transforms.Compose([
            horizontal_flip,
            convert_image_to_tensor
        ])
```

Results:

[image10]: ./README_images/iceresnet2_ADAM_scheduler_newdataaug_2ch.png
![alt_text][image10]

After three runs at a learning rate, it seems like there isn't much of a difference, though it does seem like with 2 channels, the model is 
able to fit the training data better. Nevertheless, the main thing to do now is to see if overfitting can be prevented.
Though I did do a submission using the `Ensembler.ipynb` notebook, and using an ensemble of my top 20 models I was able 
to achieve `~0.14` loss on the leaderboard.

Next experiment:
* Use a variety of weight decays

##### Experiment 7

Experiment Config:
```angular2html
{
  "model": {
    "name": "IceResNet",
    "model_configuration_name": "IceResNet",
    "parameters": {
      "num_classes": 1,
      "num_rgb": 2,
      "base": 32
    }
  },
  "optimizer": {
    "name": "ADAM",
    "parameters": {
      "lr": [0.0005],
      "weight_decay": [0.005, 0.0005, 0.00005]
    }
  },
  "scheduler": {"name": "ReduceLROnPlateau",
      "parameters": {"factor": 0.5,
       "patience": 15,
       "threshold": 0.1,
       "verbose": true
       }
  },
  "loss_function": {
    "name": "BCELoss"
  },
  "trainer_delegate": {
    "name": "StatOil"
  },
  "result_delegate": {
    "name": "StatOil"
  },
  "data_source_delegate": {
    "name": "StatOil",
    "parameters": {
      "batch_size": 40,
      "n_splits": 5,
      "splits_to_use": 5,
      "image_size": [75, 75],
      "data_handler_method": "TwoChannels",
      "testing_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/test.json",
      "training_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/train.json"
    }
  },
  "saver_delegate": {
    "name": "StatOil"
  },
  "n_epochs": 80
}


```
Data Augmentation:
```angular2html
 transforms.Compose([
            horizontal_flip,
            rotation,
            convert_image_to_tensor
        ])
```

Results:

[image11]: ./README_images/iceresnet2_ADAM_scheduler_newdataaug_2ch_rotation.png
![alt_text][image11]

It does seem like rotation is introducing more noise, though it is hard to say. Though, one thing to consider is that 
these images are already heavily preprocessed, and aren't traditional images so maybe this is the reason why it doesn't
work.

##### Experiment 8

In this experiment I tried a triple column resnet, though it didn't really fit well, never being able to generalize well 
and only reaching `~0.5` loss on the validation set.

##### Experiment 8.5

In this experiment I tried fine-tuning the pretrained DenseNet networks from torchvision, though they didn't work very 
well. Probably due to the fact that the images were being normalized between 0-1. I also tried training DenseNet from 
scratch using the new data augmentation technique (by not normalizing the data) but that also gave unstable results 
which is quite interesting as I would have imagined it would have no problem training.

##### Experiment 9

Experiment Config:
```angular2html
{
  "model": {
    "name": "IceResNet",
    "model_configuration_name": "IceResNet",
    "parameters": {
      "num_classes": 1,
      "num_rgb": 2,
      "base": 24,
      "dropout_rates": [[0,0,0,0], [0.1,0.1,0.1,0.1],[0.2,0.2,0.2,0.2]]
    }
  },
  "optimizer": {
    "name": "ADAM",
    "parameters": {
      "lr": [0.0005],
      "weight_decay": [0.00005]
    }
  },
  "scheduler": {"name": "ReduceLROnPlateau",
      "parameters": {"factor": 0.5,
       "patience": 15,
       "threshold": 0.1,
       "verbose": true
       }
  },
  "loss_function": {
    "name": "BCELoss"
  },
  "trainer_delegate": {
    "name": "StatOil"
  },
  "result_delegate": {
    "name": "StatOil"
  },
  "data_source_delegate": {
    "name": "StatOil",
    "parameters": {
      "batch_size": 40,
      "n_splits": 5,
      "splits_to_use": 5,
      "image_size": [75, 75],
      "data_handler_method": "TwoChannels",
      "testing_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/test.json",
      "training_data_path": "/home/kevin/workspace/Kaggle/Iceberg-Classifier-Challenge/src/data/train.json"
    }
  },
  "saver_delegate": {
    "name": "StatOil"
  },
  "n_epochs": 80
}
```

Data Augmentation:
```angular2html
 transforms.Compose([
            horizontal_flip,
            convert_image_to_tensor
        ])
```

Results:

[image12]: ./README_images/iceresnet2_ADAM_scheduler_newdataaug_2ch_24base.png
![alt_text][image12]

There doesn't seem to be much difference, and it seems like the minimum error I'm getting for each model is `~0.2` so 
since I've determined an neural network architecture to use, the next step is to explore ensembling with the networks.
In particular, looking at different trained models and seeing if I am able to find decent models that aren't highly
correlated with each other. So for my next submission I'm going to use a GBM with the neural networks, as the 
predictions of GBMs aren't that correlated with the NN results having a correlation of `~0.5` instead of `~0.85` 
compared with the correlation between NN's.

In my final submission by ensembling the NN's and the GBMs I got a log loss of `0.15` and though it is higher than that 
of my ensemble of NN models this was the one that I used for my final submission.
