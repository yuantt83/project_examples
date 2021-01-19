Synopsis:

This project aims to train different CNN  models to automatically
identify spiral-arm galaxies vs non-spiral galaxies, with a special
focus on whether CNN can recognise less organised high-z spirals.
This is a side project for my astrophysical work <br/>


This folder contains:<br/>

* README<br/>
* dataset/single\_prediction\_test/: galaxy images used for the training and testing. Files too
  large be copied here, so only single testing images are
  included. <br/>
  
* data_prep: notebook example for preparing the training and testing
datasets. Some astro domain knowledge is used here for feature selection.<br/>
     * sqlcl.py: SDSS python SQL module <br/>
 

* cnn\_\*.ipynb:    vanilla convolutional networks using
  Tensorflow and Keras. This is a good exercise to understand how CNN
  works and how sensitive the result is with respect to
  hyperparameters and input datasets. For our current customised dataset,  this CNN performs better than
  other transfer models. <br/>

* othermodelnames\_\*.ipynb:  transfer\_learning\_models, based on  pre-trained models (ResNet, VGG19,
Inception, EfficientNet, Yolo etc) to retrain the data. I have tried EfficientNet and VGG19 so far and the 
results are not as good as expected. They are computationally expensive and will take a while
to fine-tune the hyperparameters. More to explore. <br/>
  
* SageMaker*.ipynb: notebook examples to run  on AWS SageMaker...

* saved\_model: (selected) saved model outputs in .h5 format. (large file removed) <br/>


  

Project rights and feedback to: Dr Tiantian Yuan<br/>
www.linkedin.com/in/tiantianyuan                                                                     
