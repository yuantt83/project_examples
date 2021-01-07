Synopsis:

This project aims to train different CNN  models to automatically
identify spiral-arm galaxies vs non-spiral galaxies, with a special
focus on whether CNN can recognise less organised high-z spirals.
This is a side project for my astrophysical work <br/>


This folder contains:<br/>

* README<br/>
* dataset/single\_prediction\_test/: galaxy images used for the training and testing. Files too
  large to upload to here, so only single testing images are
  included. <br/>
  
* data_prep: notebook example for preparing the training and testing
datasets. Some domain knowledge is used here for feature selection.<br/>
     * sqlcl.py: SDSS python SQL module <br/>
 

* cnn\_\*.ipynb:    vanilla convolutional networks using
  Tensorflow and Keras. This is a good exercise to understand how CNN
  works and how sensitive the result is with respect to
  hyperparameters and input datasets. <br/>

* other\_\*.ipynb:  transfer\_learning\_models, based on  pre-trained models (ResNet, VGG19,
Inception, EfficientNet, Yolo etc) to retrain our
data. Interesting results will mostly come from using deep CNN like these. 
I will add more content here soon ! <br/>
  
* saved\_model: saved model outputs in .h5 format <br/>


  

Project rights and feedback to: Dr Tiantian Yuan<br/>
www.linkedin.com/in/tiantianyuan                                                                     
