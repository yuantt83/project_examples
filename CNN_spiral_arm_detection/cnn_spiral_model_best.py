#!/usr/bin/env python
# coding: utf-8

import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, img_to_array


# customised image sizes
szx = 256
szy = 256
szz = 3

import cv2
def random_input(img):
    shape = img.shape[:2]
    left = int(shape[0]/4)
    top = int(shape[1]/4)
    img = img[left:left*3,top:top*3,:]
    image = cv2.resize(img, shape, interpolation = cv2.INTER_CUBIC)
    image = img_to_array(image)
    return image


# In[5]:


#  Image Augmentation and generator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   rotation_range=170,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   width_shift_range=0.01,
                                   height_shift_range=0.01,
                                   fill_mode='nearest',
                                   preprocessing_function=random_input)

training_set = train_datagen.flow_from_directory('dataset/training',
                                                 target_size=(szx, szy),
                                                 batch_size=50,
                                                 shuffle=False,
                                                 seed=123,
                                                 subset='training',
                                                 class_mode='categorical')
STEP_SIZE_TRAIN = training_set.n // training_set.batch_size


# In[6]:


print(type(training_set))
print(STEP_SIZE_TRAIN)


# ### Preprocessing the Validation set

# In[7]:


valid_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.15, preprocessing_function=random_input)

valid_set = valid_datagen.flow_from_directory('dataset/training',
                                              target_size=(szx, szy),
                                              batch_size=32,
                                              subset='validation',
                                              shuffle=False,
                                              seed=123,
                                              class_mode='categorical')
STEP_SIZE_VALID = valid_set.n // valid_set.batch_size


# In[8]:


print(type(valid_set))
print(STEP_SIZE_VALID)


# ### Visualise a random training image

# In[9]:


from skimage import io

def imshow(image_RGB):
    io.imshow(image_RGB) 
    io.show()

x, y = training_set.next()
image = x[1]
imshow(image)
print(training_set.class_indices)
print(y[1])


# ## Building the CNN

# ### Initialising the CNN

# In[10]:


cnn = tf.keras.models.Sequential()


# ### Convolution

# In[11]:


# layer 1
cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=[szx, szy, szz]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# layer 2
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))
# layer 3
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))
# layer 4
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))
# layer 5
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))

# ### Flattening
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))

# ### A Dropout regularization to prevent overfitting
cnn.add(tf.keras.layers.Dropout(0.3))
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.3))
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.3))

# ### Output Layer
cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))

# ## Training the CNN
# ### Compiling the CNN

cnn.compile(optimizer=keras.optimizers.Adam(lr=0.5e-4), 
            loss = 'categorical_crossentropy', 
            metrics = ['accuracy'])


print(cnn.summary())

# from tensorflow import keras
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


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


# In[23]:


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

checkpointer = ModelCheckpoint(
    filepath = os.environ['HOME'] + '/learn_py/self/astro/dataset/wts_cnn_model_best.h5', verbose=2, save_best_only=True)

early_stopping = EarlyStopping(
    monitor = 'val_loss', patience=10, verbose=1, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=4)

csv_logger = CSVLogger(os.environ['HOME'] + '/learn_py/self/astro/dataset/train_cnn_model_best.csv')


# In[ ]:


import time
t1 = time.time()
results = cnn.fit(x = training_set, validation_data = valid_set,
                  steps_per_epoch=STEP_SIZE_TRAIN,
                  validation_steps = STEP_SIZE_VALID,
                  epochs = 50, callbacks=[plot_losses, checkpointer, early_stopping, reduce_lr, csv_logger])
t2 = time.time()
print('CNN running time is {:.2f}mins'.format((t2 - t1)/60))


# ### Save models

# In[25]:


# Save the entire model as a SavedModel.
# !mkdir -p saved_model
cnn.save('saved_model/cnn_spiral_model_best')
cnn.save('saved_model/cnn_spiral_model_best.h5')


# ### Model evaluation using Confusion Matrix and F1 score

# In[26]:


from keras.models import Model,load_model
model_check = load_model('saved_model/cnn_spiral_model_best.h5')


# In[27]:


test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=random_input)
test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size=(szx, szy),
                                            batch_size=100,
                                            shuffle=False,
                                            class_mode='categorical')

test_set.reset()

Y_pred = model_check.predict(
    test_set,
    steps=test_set.n / test_set.batch_size,
    verbose=1)

y_pred = np.argmax(Y_pred, axis=1)


# In[28]:


from sklearn.metrics import classification_report, confusion_matrix

cm=confusion_matrix(test_set.classes, y_pred)

print('The confusion matrix is \n{}\n'.format(cm))

f1 = classification_report(test_set.classes, y_pred, target_names=training_set.class_indices)
print('F1 score is {}\n'.format(f1))

# check which label is which
training_set.class_indices


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




import numpy as np
fileloc = 'dataset/single_prediction_test/nearby_large.jpg'
image_pred = image_in(fileloc)
result = cnn.predict(image_pred)

print('The predicted label is ', result)

if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical galaxy'
elif result[0][0] >= 0.5:
    prediction = 'a disk without obvious spiral structures'

print('CNN predicts that the image above is ', prediction)   


# ### Prediction 2.  
# This is also a nearby spiral galaxy. The image is taken using only one filter,  i.e., a single color image.
# Would CNN trained on 3-color images be able to recognise it as as spiral galaxy ?  Let's see.
# 
# <div>
# <img src="attachment:26da466c-ed9a-441e-aca9-e8289f0c6176.jpg" width="150"/>
# </div>
# 

# In[32]:


fileloc = 'dataset/single_prediction_test/spiral_singleband.jpg'
image_pred = image_in(fileloc)
result = cnn.predict(image_pred)

print(result)

if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical galaxy'
elif result[0][0] >= 0.5:
    prediction = 'a disk without obvious spiral structures'

print('CNN predicts that the image above is ', prediction)   


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

# In[33]:


fileloc = 'dataset/single_prediction_test/ancient_spiral_A1689B11.jpg'
image_pred = image_in(fileloc)
result = cnn.predict(image_pred)

print(result)

if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical galaxy'
elif result[0][0] >= 0.5:
    prediction = 'a disk without obvious spiral structures'

print('CNN predicts that the image above is ', prediction)   


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

# In[34]:


fileloc = 'dataset/single_prediction_test/cosmic_ring_of_fire.jpg'
image_pred = image_in(fileloc)
result = cnn.predict(image_pred)

if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical galaxy'
elif result[0][0] >= 0.5:
    prediction = 'a disk without obvious spiral structures'
print('CNN predicts that the image above is ', prediction)   



fileloc = 'dataset/single_prediction_test/clumpy1.jpg'
image_pred = image_in(fileloc)
result = cnn.predict(image_pred)
if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical galaxy'
elif result[0][0] >= 0.5:
    prediction = 'a disk without obvious spiral structures'

print(result)
print('CNN predicts that clumpy1 is ', prediction)  

fileloc = 'dataset/single_prediction_test/clumpy2.jpg'
image_pred = image_in(fileloc)
result = cnn.predict(image_pred)
if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical galaxy'
elif result[0][0] >= 0.5:
    prediction = 'a disk without obvious spiral structures'
print(result)
print('CNN predicts that clumpy2 is ', prediction)   

fileloc = 'dataset/single_prediction_test/clumpy3.jpg'
image_pred = image_in(fileloc)
result = cnn.predict(image_pred)

if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical galaxy'
elif result[0][0] >= 0.5:
    prediction = 'a disk without obvious spiral structures'

print(result)

print('CNN predicts that clumpy3 is ', prediction)  



fileloc = 'dataset/single_prediction_test/low-z_early1.jpg'
image_pred = image_in(fileloc)
result = cnn.predict(image_pred)
print(result)

if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical galaxy'
elif result[0][0] >= 0.5:
    prediction = 'a disk without obvious spiral structures'

print('CNN predicts that low-z_early1 is ', prediction)   

fileloc = 'dataset/single_prediction_test/high-z_early1.jpg'
image_pred = image_in(fileloc)
result = cnn.predict(image_pred)
if result[0][2] >= 0.5:
    prediction = 'a spiral galaxy'
elif result[0][1] >= 0.5:
    prediction = 'not a spiral galaxy but an elliptical galaxy'
elif result[0][0] >= 0.5:
    prediction = 'a disk without obvious spiral structures'
print(result)
    
print('CNN predicts that high-z_early1 is ', prediction)   




