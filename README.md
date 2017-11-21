# Kaggle-Iceberg-Classifier

### Things to do:
* Try transfer learning using ResNet/VGG

* Implement DenseNet
    * Try to use an Seq2Seq with attention on the output of the Dense blocks, and concatenate that to the 
    final feature vector before classification
    
* Try using the groups parameter in conv2d to force the network to learn representation from only 1 channel

* Read and understand the incidence angle and see if we can incorporate 
some prior information before feeding it to network

### Data augmentation/Feature Engineering
* Do some feature engineering in notebook:
    * ~~Use Sobel Kernels~~
    * ~~Use 2nd derivative~~
    * ~~Use magnitude of image~~
    * ~~Use Laplacian of the image~~
    * ~~Use Gabor filters~~
    

* ~~Do some image transformations using TorchVision~~
    This didn't seem very helpful it actually hurt more than it helped increasing log-loss on validation set from ~0.25 
    to ~0.35. Though I was only limited to using random crops and rotation. I think that there 
    would be much greater value, the reason being that the images usually have the iceberg/ship centered in the middle
    of the picture. Moreover, it is quite plausible that the training data doesn't capture all the different rotations 
    that the ships and icebergs can be in.

* ~~Standardize image~~
