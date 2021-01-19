#!/usr/bin/env python
# coding: utf-8

# # Use Convolutional Neural Network to Identify Spiral Arms
# Transfer Learning: use  VGG18
# --
# 

# ### Importing the libraries

# In[1]:


import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import tensorflow as tf
print(tf.__version__)
import keras
print(keras.__version__)


# In[2]:


from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg19
from keras.applications.vgg19 import preprocess_input


# ## Data Preprocessing
# ### Preprocessing the Training and Validation sets

# In[3]:


# customised image sizes
szx = 200
szy = 200
szz = 3

#  Image Augmentation and generator
# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    rotation_range = 170,
#                                    horizontal_flip = True,
#                                    vertical_flip=True,
#                                    width_shift_range=0.01,
#                                    height_shift_range=0.01,
#                                    fill_mode='nearest')

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


training_set = train_datagen.flow_from_directory('dataset/training',
                                                 target_size = (szx, szy),
                                                 batch_size = 32,
                                                 shuffle = True,
                                                 seed = 123,
                                                 subset='training',
                                                 class_mode = 'categorical')
STEP_SIZE_TRAIN = training_set.n // training_set.batch_size

# valid_datagen = ImageDataGenerator(rescale=1./255,
#                                    validation_split=0.15)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   validation_split=0.15)


valid_set = valid_datagen.flow_from_directory('dataset/training',
                                              target_size = (szx, szy),
                                              batch_size = 20,
                                              subset='validation',
                                              shuffle = True,
                                              seed = 123,
                                              class_mode = 'categorical')
STEP_SIZE_VALID = valid_set.n // valid_set.batch_size


# ## Setting up the VGG19 Model.
# We will  free up a  few layers to train.

# In[4]:


from keras.models import Model

base_model = vgg19.VGG19(include_top=False, weights='imagenet', 
                                     input_shape=(szx, szy, szz))


output = base_model.layers[-1].output
output = keras.layers.Flatten()(output)
model_vgg = Model(base_model.input, output)       

model_vgg.trainable = True
set_trainable = False

for layer in model_vgg.layers:
    if layer.name in ['block5_conv1', 'block4_conv1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:     
        layer.trainable = False

import pandas as pd 
pd.set_option('max_colwidth', None)
layers = [(layer, layer.name, layer.trainable) for layer in model_vgg.layers]
df_model_show = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Trainable or not'])


# In[5]:


# check that all layers are indeed frozen
df_model_show['Trainable or not'].describe()


# In[6]:


display(df_model_show.tail(20))


# In[7]:


print(model_vgg.output_shape)


# In[8]:


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

model = Sequential()
model.add(model_vgg)
model.add(Dense(512, activation='relu', input_dim=model_vgg.output_shape[1]))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.01),
              metrics=['accuracy'])

model.summary()


# ## Training the Model

# #### add visualisation to monitor the training and validation accuracy real-time:
# * code block below adapted from https://github.com/kapil-varshney/utilities/blob/master/training_plot/training_plot_ex_with_cifar10.ipynb

# In[9]:


class TrainingPlot(keras.callbacks.Callback):
    
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn-talk")
            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, linestyle=':', label = "train_loss")
            plt.plot(N, self.acc, linestyle=':', label = "train_accuracy")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_accuracy")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.show()
plot_losses = TrainingPlot()


# In[10]:


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

checkpointer = ModelCheckpoint(
    filepath='/Users/tiantianyuan/work/learn_py/self/astro/dataset/wts_vgg19_model_train.h5', verbose=2, save_best_only=True)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, verbose=1, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=4)

csv_logger = CSVLogger('/Users/tiantianyuan/work/learn_py/self/astro/dataset/wts_vgg19_model_train.csv')


# In[11]:


import time
t1 = time.time()
results = model.fit(training_set,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_set,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=3,
                    callbacks=[plot_losses, checkpointer, early_stopping, reduce_lr, csv_logger])
t2 = time.time()
print('Model running time is {:.2f}mins'.format((t2 - t1)/60))


# ### Save models

# In[12]:


# Save the entire model as a SavedModel.
# !mkdir -p saved_model
model.save('saved_model/vgg19_model_train')


# In[13]:


model.save('saved_model/vgg19_model_train.h5')


# ### Model evaluation using Confusion Matrix and F1 score

# In[14]:


from keras.models import Model,load_model
model_check = load_model('saved_model/vgg19_model_train.h5')


# In[15]:


test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size=(szx, szy),
                                            batch_size=32,
                                            shuffle=False,
                                            class_mode='categorical')

test_set.reset()

Y_pred = model_check.predict(
                            test_set,
                            steps=test_set.n / test_set.batch_size,
                            verbose=1)

y_pred = np.argmax(Y_pred, axis=1)


# In[16]:


from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(test_set.classes, y_pred)

print('The confusion matrix is \n{}\n'.format(cm))

f1 = classification_report(test_set.classes, y_pred, target_names=training_set.class_indices)
print('F1 score is {}\n'.format(f1))


# ##  Testing the CNN's performance on individual new images
# 
# ### Here is a visual representation of the training samples:
# <div>
# <img src="attachment:83b53bdd-f1ea-4d4d-99c0-1fa62942afd2.png" width="600"/>
# </div>
# 
# ### We are going to have fun with:   
# 1. A very well-known nearby large spiral galaxy with beautiful rgb color. 
# 2. A nearby spiral galaxy from a single-band color.
# 3. An ancient spiral galaxy (2.6 years after the Big Bang) that is gravitationally lensed.
# 4. A very distant 'cosmic ring of fire' galaxy
# 5. Some clumpy high-z galaxies
# 6. Some nearby and distant early-type (roundish) galaxies.

# In[ ]:


# check which label is which
training_set.class_indices


# In[ ]:


from keras.preprocessing import image
def image_in(fileloc, dimx=szx, dimy=szy):
    """ reshape a raw jpg image into an array that is acceptable by keras models.
    Parameters
    ----------
    fileloc : path and name for the input image
        an input directory under which files are searched.
    dimx, dimy : int
        should be the same as the target_size in the trained CNN
    ----------
    Return
    test_image that can be processed by kera models 
    """
    test_image = image.load_img(fileloc, target_size = (dimx, dimy))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    return test_image


# ### Prediction 1. 
# This is a beautiful nearby spiral galaxy M101. The image is much large and has more details than the training set. 
# Would CNN be able to recognise it as as spiral galaxy ?  Let's see.
# <div>
# <img src="attachment:e6bf0674-fc4e-4876-b6a1-b41ce7bb8d8f.jpg" width="150"/>
# </div>
# 

# In[ ]:


import numpy as np
fileloc = 'dataset/single_prediction_test/nearby_large.jpg'
image_pred = image_in(fileloc)
result = model_enet.predict(image_pred)

print('The predicted label is ', result)

if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but a disk'
elif result[0][0] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical'

print('model_enet predicts that the image above is ', prediction)   


# ### Prediction 2.  
# This is also a nearby spiral galaxy. The image is taken using only one filter,  i.e., a single color image.
# Would CNN trained on 3-color images be able to recognise it as as spiral galaxy ?  Let's see.
# 
# <div>
# <img src="attachment:26da466c-ed9a-441e-aca9-e8289f0c6176.jpg" width="150"/>
# </div>
# 

# In[ ]:


fileloc = 'dataset/single_prediction_test/spiral_singleband.jpg'
image_pred = image_in(fileloc)
result = model_enet.predict(image_pred)

print('The predicted label is ', result)

if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but a disk'
elif result[0][0] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical'

print('model_enet predicts that the image above is ', prediction)   


# ### Prediction 3. 
# 
# This is an ancient spiral galaxy that I studied. Would the CNN trained using 
# nearby spiral galaxies be able to identify such a proto-type spiral galaxy ? 
# https://en.wikipedia.org/wiki/A1689B11
# 
# <div>
# <img src="attachment:e08a66bc-8e35-4a29-95d9-45e44b05e81c.jpg" width="120"/>
# </div>
# 

# In[ ]:


fileloc = 'dataset/single_prediction_test/ancient_spiral_A1689B11.jpg'
image_pred = image_in(fileloc)
result = model_enet.predict(image_pred)

print('The predicted label is ', result)

if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but a disk'
elif result[0][0] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical'

print('model_enet predicts that the image above is ', prediction)   


# ### Prediction 4:  What about a weird type ? Spiral or non-Spiral ?
# This is the cosmic 'ring of fire' galaxy that I discovered. 
# https://astronomycommunity.nature.com/posts/a-distant-giant-with-a-ring-on-it
# 
# Would the CNN trained using nearby spiral galaxies classify such a ring galaxy as 'spiral' or 'nonspiral'?  I am curious.  Let's see! 
# <div>
# <img src="attachment:fd78f5c5-6af4-4ec9-a4d2-ad32d93e6dc7.jpg" width="120"/>
# </div>
# 
# 

# In[ ]:


fileloc = 'dataset/single_prediction_test/cosmic_ring_of_fire.jpg'
image_pred = image_in(fileloc)
result = model_enet.predict(image_pred)

print('The predicted label is ', result)

if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but a disk'
elif result[0][0] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical'

print('model_enet predicts that the image above is ', prediction)   


# ### Prediction 5-8:  What about some clumpy high-redshift galaxies and ellipitcals like these ? 
# [Successfully predict that they are not spirls]
# Images from the CANDELS survey:
# <div>
# <img src="attachment:90970b7f-e527-4a1b-b429-d744f4e2d663.png" width="120"/>
# <img src="attachment:91a83da5-bd63-431e-b1ed-c367b711dfd5.jpg" width="120"/>
# <img src="attachment:a70b635a-3124-4aed-8018-652496b56a94.jpg" width="120"/>
# <img src="attachment:ce4c4180-c9f5-4840-98f1-005dd9b3f1e8.png" width="120"/>
# <img src="attachment:b05db551-9142-44a6-ba20-4744e1e413cb.png" width="120"/>
# 
# </div>
# 

# In[ ]:


fileloc = 'dataset/single_prediction_test/clumpy1.jpg'
image_pred = image_in(fileloc)
result = model_enet.predict(image_pred)

print('The predicted label is ', result)

if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but a disk'
elif result[0][0] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical'

print('model_enet predicts that the image above is ', prediction)   
fileloc = 'dataset/single_prediction_test/clumpy2.jpg'
image_pred = image_in(fileloc)
result = model_enet.predict(image_pred)

print('The predicted label is ', result)

if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but a disk'
elif result[0][0] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical'

print('model_enet predicts that the image above is ', prediction)   
fileloc = 'dataset/single_prediction_test/clumpy3.jpg'
image_pred = image_in(fileloc)
image_pred = image_in(fileloc)
result = model_enet.predict(image_pred)

print('The predicted label is ', result)

if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but a disk'
elif result[0][0] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical'

print('model_enet predicts that the image above is ', prediction)   

fileloc = 'dataset/single_prediction_test/low-z_early1.jpg'
image_pred = image_in(fileloc)
image_pred = image_in(fileloc)
result = model_enet.predict(image_pred)

print('The predicted label is ', result)

if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but a disk'
elif result[0][0] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical'

print('model_enet predicts that the image above is ', prediction)   
fileloc = 'dataset/single_prediction_test/high-z_early1.jpg'
image_pred = image_in(fileloc)
image_pred = image_in(fileloc)
result = model_enet.predict(image_pred)

print('The predicted label is ', result)

if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but a disk'
elif result[0][0] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical'

print('model_enet predicts that the image above is ', prediction)   


# ### Comment
# 

# In[ ]:




