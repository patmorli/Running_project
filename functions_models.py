#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:17:47 2023

@author: patrick
"""

import numpy as np
import tensorflow as tf
import keras.applications
from keras import layers, models
from keras.models import Sequential
from keras.layers import Flatten,Conv3D
   
    
def resnet50_12channels_gpt(weights_to_use, input_resnet, dropout, final_layer_size, final_activation, dir_results_model):
    my_weights = weights_to_use
    
    # if we want to use pre-trained weights
    if weights_to_use != (None and 'imagenet'):
        modelpath = dir_results_model + weights_to_use + '.h5'
        my_weights = None # to first create a model with random weights
        print('Using saved weights from: ' + weights_to_use)
        
    # Create a ResNet50V2 model with modified input shape
    res_model_pretrained = keras.applications.ResNet50V2(include_top = False, #so that we can change input and output layer
                                        weights=my_weights, 
                                        input_tensor=input_resnet)

    # Add a global average pooling layer
    x = layers.GlobalAveragePooling2D()(res_model_pretrained.output)

    # Add a fully connected layer with 10 output nodes for classification
    x = layers.Dense(final_layer_size, activation=final_activation)(x)

    # Create a new model using the modified ResNet50V2 and fully connected layers
    model = models.Model(res_model_pretrained.input, x)
    
    if weights_to_use != (None and 'imagenet'):
        model.load_weights(modelpath)
        
    
    #model.summary()
    
    
    
    return model

def resnet50(weights_to_use, input_model, input_resnet, dropout, final_layer_size, final_activation, resnet_trainable):
    res_model_pretrained = keras.applications.ResNet50V2(include_top = False, #so that we can change input and output layer
                                        weights=weights_to_use, 
                                        input_tensor=input_resnet)
    model = Sequential()
    model.add(input_model)
    model.add(Conv3D(filters=1, kernel_size=(1,1,4), strides=(1,1,4), activation='relu')) # need bias off??
    model.add(res_model_pretrained)
    model.add(Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(final_layer_size, activation=final_activation))
    model.summary()
    
    
    
    "editing model"
    model.layers[1].trainable = resnet_trainable# trainable weights of resnet50

    # check which parts overall are frozen
    for i, layer in enumerate(model.layers):
        print(i, layer.name, "-", layer.trainable)
    
    return model

def parallel_channels_conv1D_alt(final_layer_size, final_activation, dropout, window_length, kernel_size):
    
    
    input_left_acc_x = layers.Input(shape=(window_length, 1), name = 'input_left_acc_x')
    input_left_acc_y = layers.Input(shape=(window_length, 1), name = 'input_left_acc_y')
    input_left_acc_z = layers.Input(shape=(window_length, 1), name = 'input_left_acc_z')

    conv1d_left_acc_x = layers.Conv1D(16, kernel_size, activation = 'relu', name = 'conv1d_left_acc_x')(input_left_acc_x) # output: 28x1x16
    reshape_left_acc_x = layers.Reshape((window_length - kernel_size + 1, 16, 1), name = 'reshape_left_acc_x')(conv1d_left_acc_x)
    conv2d_left_acc_x = layers.Conv2D(32, [kernel_size,3], activation = 'relu', name = 'conv2d_left_acc_x')(reshape_left_acc_x) # output: 26x14x32
    maxpool_left_acc_x = layers.MaxPool2D(5, name = 'maxpool_left_acc_x')(conv2d_left_acc_x)
    flatten_left_acc_x = layers.Flatten(name = 'flatten_left_acc_x')(maxpool_left_acc_x)

    conv1d_left_acc_y = layers.Conv1D(16, kernel_size, activation = 'relu', name = 'conv1d_left_acc_y')(input_left_acc_y) # output: 28x1x16
    reshape_left_acc_y = layers.Reshape((window_length - kernel_size + 1, 16, 1), name = 'reshape_left_acc_y')(conv1d_left_acc_y)
    conv2d_left_acc_y = layers.Conv2D(32, [kernel_size,3], activation = 'relu', name = 'conv2d_left_acc_y')(reshape_left_acc_y) # output: 26x14x32
    maxpool_left_acc_y = layers.MaxPool2D(5, name = 'maxpool_left_acc_y')(conv2d_left_acc_y)
    flatten_left_acc_y = layers.Flatten(name = 'flatten_left_acc_y')(maxpool_left_acc_y)
    
    conv1d_left_acc_z = layers.Conv1D(16, kernel_size, activation = 'relu', name = 'conv1d_left_acc_z')(input_left_acc_z) # output: 28x1x16
    reshape_left_acc_z = layers.Reshape((window_length - kernel_size + 1, 16, 1), name = 'reshape_left_acc_z')(conv1d_left_acc_z)
    conv2d_left_acc_z = layers.Conv2D(32, [kernel_size,3], activation = 'relu', name = 'conv2d_left_acc_z')(reshape_left_acc_z) # output: 26x14x32
    maxpool_left_acc_z = layers.MaxPool2D(5, name = 'maxpool_left_acc_z')(conv2d_left_acc_z) # output:
    flatten_left_acc_z = layers.Flatten(name = 'flatten_left_acc_z')(maxpool_left_acc_z)


    input_left_angvel_x = layers.Input(shape=(window_length, 1), name = 'input_left_angvel_x')
    input_left_angvel_y = layers.Input(shape=(window_length, 1), name = 'input_left_angvel_y')
    input_left_angvel_z = layers.Input(shape=(window_length, 1), name = 'input_left_angvel_z')

    conv1d_left_angvel_x = layers.Conv1D(16, kernel_size, activation = 'relu', name = 'conv1d_left_angvel_x')(input_left_angvel_x) # output: 28x1x16
    reshape_left_angvel_x = layers.Reshape((window_length - kernel_size + 1, 16, 1), name = 'reshape_left_angvel_x')(conv1d_left_angvel_x)
    conv2d_left_angvel_x = layers.Conv2D(32, [kernel_size,3], activation = 'relu', name = 'conv2d_left_angvel_x')(reshape_left_angvel_x) # output: 26x14x32
    maxpool_left_angvel_x = layers.MaxPool2D(5, name = 'maxpool_left_angvel_x')(conv2d_left_angvel_x)
    flatten_left_angvel_x = layers.Flatten(name = 'flatten_left_angvel_x')(maxpool_left_angvel_x)

    conv1d_left_angvel_y = layers.Conv1D(16, kernel_size, activation = 'relu', name = 'conv1d_left_angvel_y')(input_left_angvel_y) # output: 28x1x16
    reshape_left_angvel_y = layers.Reshape((window_length - kernel_size + 1, 16, 1), name = 'reshape_left_angvel_y')(conv1d_left_angvel_y)
    conv2d_left_angvel_y = layers.Conv2D(32, [kernel_size,3], activation = 'relu', name = 'conv2d_left_angvel_y')(reshape_left_angvel_y) # output: 26x14x32
    maxpool_left_angvel_y = layers.MaxPool2D(5, name = 'maxpool_left_angvel_y')(conv2d_left_angvel_y)
    flatten_left_angvel_y = layers.Flatten(name = 'flatten_left_angvel_y')(maxpool_left_angvel_y)
    
    conv1d_left_angvel_z = layers.Conv1D(16, kernel_size, activation = 'relu', name = 'conv1d_left_angvel_z')(input_left_angvel_z) # output: 28x1x16
    reshape_left_angvel_z = layers.Reshape((window_length - kernel_size + 1, 16, 1), name = 'reshape_left_angvel_z')(conv1d_left_angvel_z)
    conv2d_left_angvel_z = layers.Conv2D(32, [kernel_size,3], activation = 'relu', name = 'conv2d_left_angvel_z')(reshape_left_angvel_z) # output: 26x14x32
    maxpool_left_angvel_z = layers.MaxPool2D(5, name = 'maxpool_left_angvel_z')(conv2d_left_angvel_z) # output:
    flatten_left_angvel_z = layers.Flatten(name = 'flatten_left_angvel_z')(maxpool_left_angvel_z)



    input_right_acc_x = layers.Input(shape=(window_length, 1), name = 'input_right_acc_x')
    input_right_acc_y = layers.Input(shape=(window_length, 1), name = 'input_right_acc_y')
    input_right_acc_z = layers.Input(shape=(window_length, 1), name = 'input_right_acc_z')

    conv1d_right_acc_x = layers.Conv1D(16, kernel_size, activation = 'relu', name = 'conv1d_right_acc_x')(input_right_acc_x) # output: 28x1x16
    reshape_right_acc_x = layers.Reshape((window_length - kernel_size + 1, 16, 1), name = 'reshape_right_acc_x')(conv1d_right_acc_x)
    conv2d_right_acc_x = layers.Conv2D(32, [kernel_size,3], activation = 'relu', name = 'conv2d_right_acc_x')(reshape_right_acc_x) # output: 26x14x32
    maxpool_right_acc_x = layers.MaxPool2D(5, name = 'maxpool_right_acc_x')(conv2d_right_acc_x)
    flatten_right_acc_x = layers.Flatten(name = 'flatten_right_acc_x')(maxpool_right_acc_x)
    
    conv1d_right_acc_y = layers.Conv1D(16, kernel_size, activation = 'relu', name = 'conv1d_right_acc_y')(input_right_acc_y) # output: 28x1x16
    reshape_right_acc_y = layers.Reshape((window_length - kernel_size + 1, 16, 1), name = 'reshape_right_acc_y')(conv1d_right_acc_y)
    conv2d_right_acc_y = layers.Conv2D(32, [kernel_size,3], activation = 'relu', name = 'conv2d_right_acc_y')(reshape_right_acc_y) # output: 26x14x32
    maxpool_right_acc_y = layers.MaxPool2D(5, name = 'maxpool_right_acc_y')(conv2d_right_acc_y)
    flatten_right_acc_y = layers.Flatten(name = 'flatten_right_acc_y')(maxpool_right_acc_y)
    
    conv1d_right_acc_z = layers.Conv1D(16, kernel_size, activation = 'relu', name = 'conv1d_right_acc_z')(input_right_acc_z) # output: 28x1x16
    reshape_right_acc_z = layers.Reshape((window_length - kernel_size + 1, 16, 1), name = 'reshape_right_acc_z')(conv1d_right_acc_z)
    conv2d_right_acc_z = layers.Conv2D(32, [kernel_size,3], activation = 'relu', name = 'conv2d_right_acc_z')(reshape_right_acc_z) # output: 26x14x32
    maxpool_right_acc_z = layers.MaxPool2D(5, name = 'maxpool_right_acc_z')(conv2d_right_acc_z) # output:
    flatten_right_acc_z = layers.Flatten(name = 'flatten_right_acc_z')(maxpool_right_acc_z)


    input_right_angvel_x = layers.Input(shape=(window_length, 1), name = 'input_right_angvel_x')
    input_right_angvel_y = layers.Input(shape=(window_length, 1), name = 'input_right_angvel_y')
    input_right_angvel_z = layers.Input(shape=(window_length, 1), name = 'input_right_angvel_z')

    conv1d_right_angvel_x = layers.Conv1D(16, kernel_size, activation = 'relu', name = 'conv1d_right_angvel_x')(input_right_angvel_x) # output: 28x1x16
    reshape_right_angvel_x = layers.Reshape((window_length - kernel_size + 1, 16, 1), name = 'reshape_right_angvel_x')(conv1d_right_angvel_x)
    conv2d_right_angvel_x = layers.Conv2D(32, [kernel_size,3], activation = 'relu', name = 'conv2d_right_angvel_x')(reshape_right_angvel_x) # output: 26x14x32
    maxpool_right_angvel_x = layers.MaxPool2D(5, name = 'maxpool_right_angvel_x')(conv2d_right_angvel_x)
    flatten_right_angvel_x = layers.Flatten(name = 'flatten_right_angvel_x')(maxpool_right_angvel_x)

    conv1d_right_angvel_y = layers.Conv1D(16, kernel_size, activation = 'relu', name = 'conv1d_right_angvel_y')(input_right_angvel_y) # output: 28x1x16
    reshape_right_angvel_y = layers.Reshape((window_length - kernel_size + 1, 16, 1), name = 'reshape_right_angvel_y')(conv1d_right_angvel_y)
    conv2d_right_angvel_y = layers.Conv2D(32, [kernel_size,3], activation = 'relu', name = 'conv2d_right_angvel_y')(reshape_right_angvel_y) # output: 26x14x32
    maxpool_right_angvel_y = layers.MaxPool2D(5, name = 'maxpool_right_angvel_y')(conv2d_right_angvel_y)
    flatten_right_angvel_y = layers.Flatten(name = 'flatten_right_angvel_y')(maxpool_right_angvel_y)
    
    conv1d_right_angvel_z = layers.Conv1D(16, kernel_size, activation = 'relu', name = 'conv1d_right_angvel_z')(input_right_angvel_z) # output: 28x1x16
    reshape_right_angvel_z = layers.Reshape((window_length - kernel_size + 1, 16, 1), name = 'reshape_right_angvel_z')(conv1d_right_angvel_z)
    conv2d_right_angvel_z = layers.Conv2D(32, [kernel_size,3], activation = 'relu', name = 'conv2d_right_angvel_z')(reshape_right_angvel_z) # output: 26x14x32
    maxpool_right_angvel_z = layers.MaxPool2D(5, name = 'maxpool_right_angvel_z')(conv2d_right_angvel_z) # output:
    flatten_right_angvel_z = layers.Flatten(name = 'flatten_right_angvel_z')(maxpool_right_angvel_z)

     
    concat_left_right = layers.Concatenate(name = 'concat_left_right')([flatten_left_acc_x, flatten_left_acc_y, flatten_left_acc_z, flatten_left_angvel_x, flatten_left_angvel_y, flatten_left_angvel_z, flatten_right_acc_x, flatten_right_acc_y, flatten_right_acc_z, flatten_right_angvel_x, flatten_right_angvel_y, flatten_right_angvel_z]) 

    dense1 = layers.Dense(128, activation = 'relu', name = 'dense1')(concat_left_right)
    dropout_dense1 = layers.Dropout(dropout)(dense1)
    dense2 = layers.Dense(64, activation = 'relu', name = 'dense2')(dropout_dense1)
    dropout_dense2 = layers.Dropout(dropout)(dense2)
    dense3 = layers.Dense(32, activation = 'relu', name = 'dense3')(dropout_dense2)
    dropout_dense3 = layers.Dropout(dropout)(dense3)
    dense4 = layers.Dense(16, activation = 'relu', name = 'dense4')(dropout_dense3)
    dropout_dense4 = layers.Dropout(dropout)(dense4)
    output = layers.Dense(final_layer_size, activation = final_activation, name = 'output')(dropout_dense4)

    model = models.Model(inputs=[input_left_acc_x, input_left_acc_y, input_left_acc_z, input_left_angvel_x, input_left_angvel_y, input_left_angvel_z, input_right_acc_x, input_right_acc_y, input_right_acc_z, input_right_angvel_x, input_right_angvel_y, input_right_angvel_z], outputs=[output]) 

    model.summary()
    
    return model

def parallel_channels_conv1D(final_layer_size, final_activation, dropout):
    input_left_acc_x = layers.Input(shape=(5000, 1), name = 'input_left_acc_x')
    input_left_acc_y = layers.Input(shape=(5000, 1), name = 'input_left_acc_y')
    input_left_acc_z = layers.Input(shape=(5000, 1), name = 'input_left_acc_z')

    conv1d_left_acc_x = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_left_acc_x')(input_left_acc_x) # output: 28x1x16
    reshape_left_acc_x = layers.Reshape((4501, 16, 1), name = 'reshape_left_acc_x')(conv1d_left_acc_x)
    conv2d_left_acc_x = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_left_acc_x')(reshape_left_acc_x) # output: 26x14x32
    maxpool_left_acc_x = layers.MaxPool2D(5, name = 'maxpool_left_acc_x')(conv2d_left_acc_x)
    flatten_left_acc_x = layers.Flatten(name = 'flatten_left_acc_x')(maxpool_left_acc_x)

    conv1d_left_acc_y = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_left_acc_y')(input_left_acc_y) # output: 28x1x16
    reshape_left_acc_y = layers.Reshape((4501, 16, 1), name = 'reshape_left_acc_y')(conv1d_left_acc_y)
    conv2d_left_acc_y = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_left_acc_y')(reshape_left_acc_y) # output: 26x14x32
    maxpool_left_acc_y = layers.MaxPool2D(5, name = 'maxpool_left_acc_y')(conv2d_left_acc_y)
    flatten_left_acc_y = layers.Flatten(name = 'flatten_left_acc_y')(maxpool_left_acc_y)
    
    conv1d_left_acc_z = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_left_acc_z')(input_left_acc_z) # output: 28x1x16
    reshape_left_acc_z = layers.Reshape((4501, 16, 1), name = 'reshape_left_acc_z')(conv1d_left_acc_z)
    conv2d_left_acc_z = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_left_acc_z')(reshape_left_acc_z) # output: 26x14x32
    maxpool_left_acc_z = layers.MaxPool2D(5, name = 'maxpool_left_acc_z')(conv2d_left_acc_z) # output:
    flatten_left_acc_z = layers.Flatten(name = 'flatten_left_acc_z')(maxpool_left_acc_z)


    input_left_angvel_x = layers.Input(shape=(5000, 1), name = 'input_left_angvel_x')
    input_left_angvel_y = layers.Input(shape=(5000, 1), name = 'input_left_angvel_y')
    input_left_angvel_z = layers.Input(shape=(5000, 1), name = 'input_left_angvel_z')

    conv1d_left_angvel_x = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_left_angvel_x')(input_left_angvel_x) # output: 28x1x16
    reshape_left_angvel_x = layers.Reshape((4501, 16, 1), name = 'reshape_left_angvel_x')(conv1d_left_angvel_x)
    conv2d_left_angvel_x = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_left_angvel_x')(reshape_left_angvel_x) # output: 26x14x32
    maxpool_left_angvel_x = layers.MaxPool2D(5, name = 'maxpool_left_angvel_x')(conv2d_left_angvel_x)
    flatten_left_angvel_x = layers.Flatten(name = 'flatten_left_angvel_x')(maxpool_left_angvel_x)

    conv1d_left_angvel_y = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_left_angvel_y')(input_left_angvel_y) # output: 28x1x16
    reshape_left_angvel_y = layers.Reshape((4501, 16, 1), name = 'reshape_left_angvel_y')(conv1d_left_angvel_y)
    conv2d_left_angvel_y = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_left_angvel_y')(reshape_left_angvel_y) # output: 26x14x32
    maxpool_left_angvel_y = layers.MaxPool2D(5, name = 'maxpool_left_angvel_y')(conv2d_left_angvel_y)
    flatten_left_angvel_y = layers.Flatten(name = 'flatten_left_angvel_y')(maxpool_left_angvel_y)
    
    conv1d_left_angvel_z = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_left_angvel_z')(input_left_angvel_z) # output: 28x1x16
    reshape_left_angvel_z = layers.Reshape((4501, 16, 1), name = 'reshape_left_angvel_z')(conv1d_left_angvel_z)
    conv2d_left_angvel_z = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_left_angvel_z')(reshape_left_angvel_z) # output: 26x14x32
    maxpool_left_angvel_z = layers.MaxPool2D(5, name = 'maxpool_left_angvel_z')(conv2d_left_angvel_z) # output:
    flatten_left_angvel_z = layers.Flatten(name = 'flatten_left_angvel_z')(maxpool_left_angvel_z)



    input_right_acc_x = layers.Input(shape=(5000, 1), name = 'input_right_acc_x')
    input_right_acc_y = layers.Input(shape=(5000, 1), name = 'input_right_acc_y')
    input_right_acc_z = layers.Input(shape=(5000, 1), name = 'input_right_acc_z')

    conv1d_right_acc_x = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_right_acc_x')(input_right_acc_x) # output: 28x1x16
    reshape_right_acc_x = layers.Reshape((4501, 16, 1), name = 'reshape_right_acc_x')(conv1d_right_acc_x)
    conv2d_right_acc_x = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_right_acc_x')(reshape_right_acc_x) # output: 26x14x32
    maxpool_right_acc_x = layers.MaxPool2D(5, name = 'maxpool_right_acc_x')(conv2d_right_acc_x)
    flatten_right_acc_x = layers.Flatten(name = 'flatten_right_acc_x')(maxpool_right_acc_x)
    
    conv1d_right_acc_y = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_right_acc_y')(input_right_acc_y) # output: 28x1x16
    reshape_right_acc_y = layers.Reshape((4501, 16, 1), name = 'reshape_right_acc_y')(conv1d_right_acc_y)
    conv2d_right_acc_y = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_right_acc_y')(reshape_right_acc_y) # output: 26x14x32
    maxpool_right_acc_y = layers.MaxPool2D(5, name = 'maxpool_right_acc_y')(conv2d_right_acc_y)
    flatten_right_acc_y = layers.Flatten(name = 'flatten_right_acc_y')(maxpool_right_acc_y)
    
    conv1d_right_acc_z = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_right_acc_z')(input_right_acc_z) # output: 28x1x16
    reshape_right_acc_z = layers.Reshape((4501, 16, 1), name = 'reshape_right_acc_z')(conv1d_right_acc_z)
    conv2d_right_acc_z = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_right_acc_z')(reshape_right_acc_z) # output: 26x14x32
    maxpool_right_acc_z = layers.MaxPool2D(5, name = 'maxpool_right_acc_z')(conv2d_right_acc_z) # output:
    flatten_right_acc_z = layers.Flatten(name = 'flatten_right_acc_z')(maxpool_right_acc_z)


    input_right_angvel_x = layers.Input(shape=(5000, 1), name = 'input_right_angvel_x')
    input_right_angvel_y = layers.Input(shape=(5000, 1), name = 'input_right_angvel_y')
    input_right_angvel_z = layers.Input(shape=(5000, 1), name = 'input_right_angvel_z')

    conv1d_right_angvel_x = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_right_angvel_x')(input_right_angvel_x) # output: 28x1x16
    reshape_right_angvel_x = layers.Reshape((4501, 16, 1), name = 'reshape_right_angvel_x')(conv1d_right_angvel_x)
    conv2d_right_angvel_x = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_right_angvel_x')(reshape_right_angvel_x) # output: 26x14x32
    maxpool_right_angvel_x = layers.MaxPool2D(5, name = 'maxpool_right_angvel_x')(conv2d_right_angvel_x)
    flatten_right_angvel_x = layers.Flatten(name = 'flatten_right_angvel_x')(maxpool_right_angvel_x)

    conv1d_right_angvel_y = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_right_angvel_y')(input_right_angvel_y) # output: 28x1x16
    reshape_right_angvel_y = layers.Reshape((4501, 16, 1), name = 'reshape_right_angvel_y')(conv1d_right_angvel_y)
    conv2d_right_angvel_y = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_right_angvel_y')(reshape_right_angvel_y) # output: 26x14x32
    maxpool_right_angvel_y = layers.MaxPool2D(5, name = 'maxpool_right_angvel_y')(conv2d_right_angvel_y)
    flatten_right_angvel_y = layers.Flatten(name = 'flatten_right_angvel_y')(maxpool_right_angvel_y)
    
    conv1d_right_angvel_z = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_right_angvel_z')(input_right_angvel_z) # output: 28x1x16
    reshape_right_angvel_z = layers.Reshape((4501, 16, 1), name = 'reshape_right_angvel_z')(conv1d_right_angvel_z)
    conv2d_right_angvel_z = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_right_angvel_z')(reshape_right_angvel_z) # output: 26x14x32
    maxpool_right_angvel_z = layers.MaxPool2D(5, name = 'maxpool_right_angvel_z')(conv2d_right_angvel_z) # output:
    flatten_right_angvel_z = layers.Flatten(name = 'flatten_right_angvel_z')(maxpool_right_angvel_z)

     
    concat_left_right = layers.Concatenate(name = 'concat_left_right')([flatten_left_acc_x, flatten_left_acc_y, flatten_left_acc_z, flatten_left_angvel_x, flatten_left_angvel_y, flatten_left_angvel_z, flatten_right_acc_x, flatten_right_acc_y, flatten_right_acc_z, flatten_right_angvel_x, flatten_right_angvel_y, flatten_right_angvel_z]) 

    dense1 = layers.Dense(128, activation = 'relu', name = 'dense1')(concat_left_right)
    dropout_dense1 = layers.Dropout(dropout)(dense1)
    dense2 = layers.Dense(64, activation = 'relu', name = 'dense2')(dropout_dense1)
    dropout_dense2 = layers.Dropout(dropout)(dense2)
    dense3 = layers.Dense(32, activation = 'relu', name = 'dense3')(dropout_dense2)
    dropout_dense3 = layers.Dropout(dropout)(dense3)
    dense4 = layers.Dense(16, activation = 'relu', name = 'dense4')(dropout_dense3)
    dropout_dense4 = layers.Dropout(dropout)(dense4)
    output = layers.Dense(final_layer_size, activation = final_activation, name = 'output')(dropout_dense4)

    model = models.Model(inputs=[input_left_acc_x, input_left_acc_y, input_left_acc_z, input_left_angvel_x, input_left_angvel_y, input_left_angvel_z, input_right_acc_x, input_right_acc_y, input_right_acc_z, input_right_angvel_x, input_right_angvel_y, input_right_angvel_z], outputs=[output]) 

    model.summary()
    return model


def parallel_channels_LSTM(final_layer_size, final_activation, dropout):
    
    input_left_acc_x = layers.Input(shape=(5000, 1), name = 'input_left_acc_x')
    input_left_acc_y = layers.Input(shape=(5000, 1), name = 'input_left_acc_y')
    #input_left_acc_z = layers.Input(shape=(5000, 1), name = 'input_left_acc_z')

    LSTM1_left_acc_x = layers.LSTM(64, activation = 'relu', return_sequences = True, name = 'LSTM1_left_acc_x')(input_left_acc_x) # output: 28x1x16
    LSTM2_left_acc_x = layers.LSTM(32, activation = 'relu', return_sequences = False, name = 'LSTM_2left_acc_x')(LSTM1_left_acc_x) # output: 26x14x32

    LSTM1_left_acc_y = layers.LSTM(64, activation = 'relu', return_sequences = True, name = 'LSTM1_left_acc_y')(input_left_acc_y) # output: 28x1x16
    LSTM2_left_acc_y = layers.LSTM(32, activation = 'relu', return_sequences = False, name = 'LSTM_2left_acc_y')(LSTM1_left_acc_y) # output: 26x14x32
    
    
    """
    conv1d_left_acc_z = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_left_acc_z')(input_left_acc_z) # output: 28x1x16
    reshape_left_acc_z = layers.Reshape((4501, 16, 1), name = 'reshape_left_acc_z')(conv1d_left_acc_z)
    conv2d_left_acc_z = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_left_acc_z')(reshape_left_acc_z) # output: 26x14x32
    maxpool_left_acc_z = layers.MaxPool2D(5, name = 'maxpool_left_acc_z')(conv2d_left_acc_z) # output:
    flatten_left_acc_z = layers.Flatten(name = 'flatten_left_acc_z')(maxpool_left_acc_z)


    input_left_angvel_x = layers.Input(shape=(5000, 1), name = 'input_left_angvel_x')
    input_left_angvel_y = layers.Input(shape=(5000, 1), name = 'input_left_angvel_y')
    input_left_angvel_z = layers.Input(shape=(5000, 1), name = 'input_left_angvel_z')

    conv1d_left_angvel_x = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_left_angvel_x')(input_left_angvel_x) # output: 28x1x16
    reshape_left_angvel_x = layers.Reshape((4501, 16, 1), name = 'reshape_left_angvel_x')(conv1d_left_angvel_x)
    conv2d_left_angvel_x = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_left_angvel_x')(reshape_left_angvel_x) # output: 26x14x32
    maxpool_left_angvel_x = layers.MaxPool2D(5, name = 'maxpool_left_angvel_x')(conv2d_left_angvel_x)
    flatten_left_angvel_x = layers.Flatten(name = 'flatten_left_angvel_x')(maxpool_left_angvel_x)

    conv1d_left_angvel_y = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_left_angvel_y')(input_left_angvel_y) # output: 28x1x16
    reshape_left_angvel_y = layers.Reshape((4501, 16, 1), name = 'reshape_left_angvel_y')(conv1d_left_angvel_y)
    conv2d_left_angvel_y = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_left_angvel_y')(reshape_left_angvel_y) # output: 26x14x32
    maxpool_left_angvel_y = layers.MaxPool2D(5, name = 'maxpool_left_angvel_y')(conv2d_left_angvel_y)
    flatten_left_angvel_y = layers.Flatten(name = 'flatten_left_angvel_y')(maxpool_left_angvel_y)
    
    conv1d_left_angvel_z = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_left_angvel_z')(input_left_angvel_z) # output: 28x1x16
    reshape_left_angvel_z = layers.Reshape((4501, 16, 1), name = 'reshape_left_angvel_z')(conv1d_left_angvel_z)
    conv2d_left_angvel_z = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_left_angvel_z')(reshape_left_angvel_z) # output: 26x14x32
    maxpool_left_angvel_z = layers.MaxPool2D(5, name = 'maxpool_left_angvel_z')(conv2d_left_angvel_z) # output:
    flatten_left_angvel_z = layers.Flatten(name = 'flatten_left_angvel_z')(maxpool_left_angvel_z)



    input_right_acc_x = layers.Input(shape=(5000, 1), name = 'input_right_acc_x')
    input_right_acc_y = layers.Input(shape=(5000, 1), name = 'input_right_acc_y')
    input_right_acc_z = layers.Input(shape=(5000, 1), name = 'input_right_acc_z')

    conv1d_right_acc_x = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_right_acc_x')(input_right_acc_x) # output: 28x1x16
    reshape_right_acc_x = layers.Reshape((4501, 16, 1), name = 'reshape_right_acc_x')(conv1d_right_acc_x)
    conv2d_right_acc_x = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_right_acc_x')(reshape_right_acc_x) # output: 26x14x32
    maxpool_right_acc_x = layers.MaxPool2D(5, name = 'maxpool_right_acc_x')(conv2d_right_acc_x)
    flatten_right_acc_x = layers.Flatten(name = 'flatten_right_acc_x')(maxpool_right_acc_x)
    
    conv1d_right_acc_y = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_right_acc_y')(input_right_acc_y) # output: 28x1x16
    reshape_right_acc_y = layers.Reshape((4501, 16, 1), name = 'reshape_right_acc_y')(conv1d_right_acc_y)
    conv2d_right_acc_y = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_right_acc_y')(reshape_right_acc_y) # output: 26x14x32
    maxpool_right_acc_y = layers.MaxPool2D(5, name = 'maxpool_right_acc_y')(conv2d_right_acc_y)
    flatten_right_acc_y = layers.Flatten(name = 'flatten_right_acc_y')(maxpool_right_acc_y)
    
    conv1d_right_acc_z = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_right_acc_z')(input_right_acc_z) # output: 28x1x16
    reshape_right_acc_z = layers.Reshape((4501, 16, 1), name = 'reshape_right_acc_z')(conv1d_right_acc_z)
    conv2d_right_acc_z = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_right_acc_z')(reshape_right_acc_z) # output: 26x14x32
    maxpool_right_acc_z = layers.MaxPool2D(5, name = 'maxpool_right_acc_z')(conv2d_right_acc_z) # output:
    flatten_right_acc_z = layers.Flatten(name = 'flatten_right_acc_z')(maxpool_right_acc_z)


    input_right_angvel_x = layers.Input(shape=(5000, 1), name = 'input_right_angvel_x')
    input_right_angvel_y = layers.Input(shape=(5000, 1), name = 'input_right_angvel_y')
    input_right_angvel_z = layers.Input(shape=(5000, 1), name = 'input_right_angvel_z')

    conv1d_right_angvel_x = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_right_angvel_x')(input_right_angvel_x) # output: 28x1x16
    reshape_right_angvel_x = layers.Reshape((4501, 16, 1), name = 'reshape_right_angvel_x')(conv1d_right_angvel_x)
    conv2d_right_angvel_x = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_right_angvel_x')(reshape_right_angvel_x) # output: 26x14x32
    maxpool_right_angvel_x = layers.MaxPool2D(5, name = 'maxpool_right_angvel_x')(conv2d_right_angvel_x)
    flatten_right_angvel_x = layers.Flatten(name = 'flatten_right_angvel_x')(maxpool_right_angvel_x)

    conv1d_right_angvel_y = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_right_angvel_y')(input_right_angvel_y) # output: 28x1x16
    reshape_right_angvel_y = layers.Reshape((4501, 16, 1), name = 'reshape_right_angvel_y')(conv1d_right_angvel_y)
    conv2d_right_angvel_y = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_right_angvel_y')(reshape_right_angvel_y) # output: 26x14x32
    maxpool_right_angvel_y = layers.MaxPool2D(5, name = 'maxpool_right_angvel_y')(conv2d_right_angvel_y)
    flatten_right_angvel_y = layers.Flatten(name = 'flatten_right_angvel_y')(maxpool_right_angvel_y)
    
    conv1d_right_angvel_z = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_right_angvel_z')(input_right_angvel_z) # output: 28x1x16
    reshape_right_angvel_z = layers.Reshape((4501, 16, 1), name = 'reshape_right_angvel_z')(conv1d_right_angvel_z)
    conv2d_right_angvel_z = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_right_angvel_z')(reshape_right_angvel_z) # output: 26x14x32
    maxpool_right_angvel_z = layers.MaxPool2D(5, name = 'maxpool_right_angvel_z')(conv2d_right_angvel_z) # output:
    flatten_right_angvel_z = layers.Flatten(name = 'flatten_right_angvel_z')(maxpool_right_angvel_z)
    """
     
    concat_left_right = layers.Concatenate(name = 'concat_left_right')([LSTM2_left_acc_x, LSTM2_left_acc_y]) 

    dense1 = layers.Dense(128, activation = 'relu', name = 'dense1')(concat_left_right)
    dropout_dense1 = layers.Dropout(dropout)(dense1)
    dense2 = layers.Dense(64, activation = 'relu', name = 'dense2')(dropout_dense1)
    dropout_dense2 = layers.Dropout(dropout)(dense2)
    dense3 = layers.Dense(32, activation = 'relu', name = 'dense3')(dropout_dense2)
    dropout_dense3 = layers.Dropout(dropout)(dense3)
    dense4 = layers.Dense(16, activation = 'relu', name = 'dense4')(dropout_dense3)
    dropout_dense4 = layers.Dropout(dropout)(dense4)
    output = layers.Dense(final_layer_size, activation = final_activation, name = 'output')(dropout_dense4)

    model = models.Model(inputs=[input_left_acc_x, input_left_acc_y], outputs=[output]) 

    model.summary()
    return model



 
    
    