#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import tensorflow, keras libraries 
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# load the mnist dataset provided by Keras 
mnist = keras.datasets.mnist

# load the training and test data
(img_rows, img_cols) = (28,28)
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# lets plot some data samples 
plt.figure(figsize=(10,10)) 
for i in range(25):
    plt.subplot(5,5,i+1) 
    plt.imshow(x_train[i], cmap=plt.cm.gray) 
    plt.xlabel(y_test[i])
plt.show()


# In[4]:


#Normalization of Training and Test Data to Learn Faster
from keras.utils import to_categorical

# one hot encode the results
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)

# see the dimensions of the data 
print('Training X dimensions: ', x_train.shape) 
print('Training Y dimensions: ', y_train.shape) 
print('Testing X dimensions: ', x_test.shape) 
print('Testing Y dimensions: ', y_test.shape)

# normalize the data to values between 0 and 1 
x_train, x_test = x_train / 255.0, x_test / 255.0


# In[5]:


#Neural Network Code
from keras.models import Sequential
from keras.layers import Dense, Flatten

# build a simple Neural Network
model = Sequential() 
model.add(Flatten(input_shape=(28, 28))) 
model.add(Dense(512, activation='relu')) 
model.add(Dense(10, activation='softmax'))

# show summary 
model.summary()

# assign the optimizer for the model and define loss function 
model.compile(optimizer='adam',
loss='categorical_crossentropy', metrics=['accuracy'])

# run the actual training
history = model.fit(x_train, y_train, epochs=1, validation_split=0.33)

# evaluate on test data 
model.evaluate(x_test, y_test)


# In[ ]:




