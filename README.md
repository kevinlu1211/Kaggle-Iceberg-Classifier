## Kaggle Iceberg Classifier 

This repository is a competition hosted on [Kaggle](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge) for 
detecting whether the picture has a iceberg or a ship.

## Experiment Framework

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

### DenseNet121 results for different hyperparameters
![alt text][image2]


