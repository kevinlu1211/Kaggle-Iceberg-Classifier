# Kaggle Iceberg Classifier 


#### High Priority
- [ ] Plan/Implement training/testing pipeline for ensembling networks
    - [ ] Read up about Gang of Four Patterns
    - [ ] Create UML diagram for pipeline
    - [ ] Should probably write an interface that the forces the network to conform to some output format
    
- [ ] Implement pipeline for K-fold cross validation

- [ ] Do some research in regards to how best to store hyperparameter configs


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

