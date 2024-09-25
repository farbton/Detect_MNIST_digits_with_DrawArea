# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:37:34 2022
@author: Kirko
Lückencode für 2. Versuch im Praktikum Maschinelles Lernen zur Klassifizierung 
von handschriftlichen Zahlen
MNIST-Zahlen laden, CNN-Model trainieren, Image vorhersagen, Ausgabe plotten
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,\
    Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, metrics
from sklearn.metrics import ConfusionMatrixDisplay

# normalize all train_images and test_imnages around zero
def normalize_train_test(train_img, test_img):
    erwart_train = np.mean(train_img)
    standard_train = np.std(train_img)
    train_img = (train_img-erwart_train)/standard_train
    
    erwart_test = np.mean(test_img)
    standard_test = np.std(test_img)
    test_img = (test_img-erwart_test)/standard_test
    return train_img, test_img

# # load train and test data
(train_images, train_labels), (test_images, test_labels) = ...

# normalize the images between 0 and 1.
train_images = ...
test_images  = ...

# normalize all images around zero
train_images, test_images = normalize_train_test(train_images, test_images)

# Reshape the images because keras requires three dim (28, 28, 1)
train_images = np.expand_dims(train_images, axis=3)
test_images  = np.expand_dims(test_images, axis=3)

print("train_images.shape: " + str(train_images.shape))
print("train_labels.shape: " + str(train_labels.shape))

#Vorhersage plotten
def plot_predict(label):
  ...
  pass
  
train = 0
if train == 1:
    # define variables
    NUM_FILTER  = ...
    FILTER_SIZE = ...
    EPOCHS      = ...
    
    #define modelarchitecture 
    model = Sequential([
      ...
      
    ])
   
    # show network architecture
    model.summary()
    
    reduce_lr = ReduceLROnPlateau(monitor  = 'val_loss', 
                                  factor   = ...,
                                  patience = ..., 
                                  min_lr   = ...)
    
    # compile the model
    model.compile(
      optimizer = optimizers.Adam(learning_rate=1e-3),
                  loss      = 'categorical_crossentropy',
                  metrics   = ['accuracy'],
    )
    
    callbacks = [
        TensorBoard(
            log_dir         = "log_dir",
            histogram_freq  = 1,
            embeddings_freq = 1,
        ),
        reduce_lr,
    ]

    #train the model
    H = model.fit(
      ...
      )
    
    model.save("PML_MNIST_CNN_StudentName")
    print(H.params)
    
    # plot the model training history
    def plot_train():
        ...
        pass
    plot_train()
    
else:
    model = load_model("PML_MNIST_CNN_StudentName") #
    one_or_all = 1
    
    #predict one image and plot result
    if one_or_all == 1:  
        ...
        pass
     
    # predict whole test-dataset with confusion matrix
    else:
        ...
        pass
    
      
    



