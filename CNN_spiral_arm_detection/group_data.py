#!/usr/bin/env python
# coding: utf-8

# # pre-processing data
# data source: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data?select=images_test_rev1.zip
# 
# Downloaded data summary:
# * 61578 jpg images in the images_training_rev1 folder. 
# * 79975 jpg images in the images_test_rev1 folder. 
# * training_solutions_rev1.csv contains meta info about the dataset.
# 
# What we need:
# 
# This project has a different and simpler goal than the original Kaggle competition.
# We are going to  use this dataste train a NN to identify spiral arms only. The final goal
# is to let the NN tell us whether an image contains spiral arms. We do not care about the rest of the galaxy morphology. 
# 
# Hence our data processing plan is:
# 1. Select galaxies that have spiral arms in training and test sets.  We can use Class4.1 > 0.5 in training_solutions_rev1.csv as a criterion.
# 2. Put galaxies with Class4.1 >= 0.5 into a new folder called spirals.
# 3. Put galaxies with 0 < Class4.1 < 0.5 into a new folder called non-spirals.
# [optional later step]4. Put galaxies with Class4.1 = 0 into a new folder called rounds. 
# 
# 
# 

# ### load training data meta information

# In[4]:


import pandas as pd
meta_info = pd.read_csv('training_solutions_rev1.csv')
display(meta_info.head())


# In[5]:


# check datatypes of each columnn
print(meta_info.info())


# In[15]:


# select spiral galaxies for columns  ID
meta_use_spiral = meta_info[meta_info['Class4.1'] >= 0.5].iloc[:, 0].values
# meta_use[:, [0]].astype(int)


# In[16]:


print(type(meta_use_spiral))
print(meta_use_spiral.shape)
print(meta_use_spiral)


# In[28]:


import os
import shutil
import numpy as np

path_root = './dataset/images_training_rev1/'
write_dir_name = './dataset/training/spirals/'

for num in meta_use_spiral:
    file = path_root + str(num) + '.jpg'
    if os.path.isfile(file):
        shutil.copy(file, write_dir_name, follow_symlinks=True)


# now we have 10397 spirals in the training/spirals folder

# In[34]:


cond = (meta_info['Class4.1'] != 0) & (meta_info['Class4.1'] < 0.5)
meta_use_nonspiral = meta_info[cond].iloc[:, 0].values


# In[36]:


write_dir_name = './dataset/training/nonspirals/'

# nonspirals are 5 times more. To save time, we just need similar number of non-spirals
for num in meta_use_nonspiral[0: 10400]:
    file = path_root + str(num) + '.jpg'
    if os.path.isfile(file):
        shutil.copy(file, write_dir_name, follow_symlinks=True)


# In[39]:


# Test set spirals
path_root = './dataset/images_training_rev1/'
write_dir_name = './dataset/test/spirals/'
for num in meta_use_spiral[0:5000]:
    file = path_root + str(num) + '.jpg'
    if os.path.isfile(file):
        shutil.copy(file, write_dir_name, follow_symlinks=True)

# Test set non spirals
path_root = './dataset/images_training_rev1/'
write_dir_name = './dataset/test/nonspirals/'
for num in meta_use_nonspiral[10400: 15400]:
    file = path_root + str(num) + '.jpg'
    if os.path.isfile(file):
        shutil.copy(file, write_dir_name, follow_symlinks=True)


# In[40]:


print('now we have {} spirals in the train/spiral folder'.format(10397))
print('now we have {} spirals in the train/nonspiral folder'.format(10400))
print('now we have {} spirals in the test/spiral folder'.format(5000))
print('now we have {} spirals in the test/nonspiral folder'.format(5000))


# In[ ]:




