# Kaggle Iceberg Classifier 

This repository is a competition hosted on [Kaggle](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge) for 
detecting whether the picture has a iceberg or a ship.

Currently following a [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html) branching model, so 
features being actively developed are on the `origin/features` branches

#### Experiment Framework
This is framework in the `Experiment` folder that allows for faster iterations on training and optimizing neural network 
architectures. This is how it works:

1. Provide a configuration file which is in JSON format, this file has all the hyperparameters for the model, the batch
size being used, basically, all the things that are configurable are stored here. An example can be found in
`experiment_config/experiment_1/experiment_1_qsnet.json` but for convenience this is what is looks like:
```
{
  "id": "qsnet",
  "model_name": "QSNet",
  "model_parameters": {
      "dropout_rate": 0.5
  },
  "optimizer_name": "ADAM",
  "optimizer_parameters": {
      "lr": 0.00005,
      "weight_decay": 0.00005
  },
  "loss_function_name": "BCELoss",
  "loss_function_parameters": {},
  "output_transformation_name": "sigmoid",
  "trainer_delegate_name": "QSNet",
  "data_source_delegate_name": "QSNet",
  "data_source_delegate_parameters": {
    "batch_size": 64
  },
  "n_epochs": 5
}
```

2. The ExperimentFactory located in `Experiment/ExperimentFactory` reads these JSON configuration files and creates
an experiment which has all the information that is needed to run a training process end-to-end. This is done by using 
the files in `experiment_mappings` which provides a lookup for the model architecture, the trainer and data source 
delegates, and anything else that is configurable in the experiment. For example in `experiment_mappings/models` there 
is a dictionary that has a key value pair `"QSNet": <reference to QSNet>"`and similarly for the trainer, 
and data source delegates.

3. The data source, and trainer delegate has various hooks during the data preprocessing and training process which can 
be found in the `Experiment/AbstractDataSourceDelegate` and `Experiment/AbstractTrainerDelegate` classes


#### High Priority
    
- [ ] Implement a framework for easier optimization of various neural networks architectures, random parameter 
search through genetic algorithm search and easy ensembling

    - [X] Implement the part of the framework which has the core of the training process
    - [X] Implement the part of the framework which has hooks into the training process
    - [X] Implement the hooks for K-fold cross validation
    - [X] Do some research in regards to how best to store hyperparameter configs
    - [ ] Update framework README   
    - [ ] Implement Breeder class
    - [ ] Clean up the hooks in TrainerDelegate, as a lot of them aren't needed 


#### Medium Priority
- [ ] Try transfer learning using ResNet/VGG

- [ ] Try Squeeze Excitation Nets

 - [X] Implement [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
    - [ ] Try to use an Seq2Seq with attention on the output of the Dense blocks, and concatenate that to the 
    flattened feature vector before classification. Or even simpler, just concatentate the feature vectors of 
    the Dense block to the flattened feature layer
    
- [ ]  Try using the groups parameter in conv2d to force the network to learn representation from only 1 channel, and then 
concatentate that representation with the ones that use all the channels

- [ ] Read and understand the incidence angle and see if we can incorporate 
some prior information before feeding it to network

- [ ] Try using more engineered features such as Wavelets


#### Low Priority
* Find some way to use the large amount of test data that is available, maybe use a semi-supervised VAE to do the 
labeling?

#### Easy things to do
- [X] Do some feature engineering in notebook:
    - [X] Use Sobel Kernels
    - [X] Use 2nd derivative
    - [X] Use magnitude of image
    - [X] Use Laplacian of the image
    - [X] Use Gabor filters
    
- [ ] Do some data augmentation in notebook:   
    - [ ] Implement custom flipping/rotations for images  
    - [ ] Try to use or implement the 5 crop data augmentation technique that torch vision has
    
- [X] Use image transformations/standardization in torchvision library

    * This didn't seem very helpful it actually hurt more than it helped increasing log-loss on validation set from ~0.25 
    to ~0.35. Though I was only limited to using random crops and rotation. I think that there 
    would be much greater value, the reason being that the images usually have the iceberg/ship centered in the middle
    of the picture. Moreover, it is quite plausible that the training data doesn't capture all the different rotations 
    that the ships and icebergs can be in.

