#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:33:20 2022

@author: patrick

helpful resource: https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b

"""

import keras
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM, Dense,Flatten,Conv3D
from tensorflow.keras.optimizers import Adam

def create_my_model_resnet50_cont(input_my_model, input_resnet, weights_to_use, dropout, n_bins):
    # resnet50 with weights from "imagenet" dataset
    res_model_pretrained = keras.applications.ResNet50(include_top = False, #so that we can change input and output layer
                                            weights=weights_to_use, 
                                            input_tensor=input_resnet)
    my_model = Sequential()
    my_model.add(input_my_model)
    my_model.add(Conv3D(filters=1, kernel_size=(1,1,4), strides=(1,1,4)), activation = 'relu') # need bias off??
    my_model.add(res_model_pretrained)
    my_model.add(Flatten())
    my_model.add(layers.Dropout(dropout))
    my_model.add(layers.Dense(1, activation='relu'))
    my_model.summary()
    
    my_model.compile(loss='categorical_crossentropy',
                     optimizer=keras.optimizers.RMSprop(learning_rate=2e-5),
                     metrics=['categorical_crossentropy'])
    
    return my_model

def create_my_model_resnet50_class(input_my_model, input_resnet, weights_to_use, dropout, n_bins, learning_rate):
    # resnet50 with weights from "imagenet" dataset
    res_model_pretrained = keras.applications.ResNet50(include_top = False, #so that we can change input and output layer
                                            weights=weights_to_use, 
                                            input_tensor=input_resnet)
    my_model = Sequential()
    my_model.add(input_my_model)
    my_model.add(Conv3D(filters=1, kernel_size=(1,1,4), strides=(1,1,4), activation='relu')) # need bias off??
    my_model.add(res_model_pretrained)
    my_model.add(Flatten())
    my_model.add(layers.Dropout(dropout))
    my_model.add(layers.Dense(n_bins, activation='softmax'))
    my_model.summary()
    
    my_model.compile(loss='categorical_crossentropy',
                     optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                     metrics=['categorical_crossentropy'])
    
    return my_model

def create_my_model_resnet50_class_more_conv_layers(input_my_model, input_resnet, weights_to_use, dropout, n_bins,learning_rate):
    # resnet50 with weights from "imagenet" dataset
    res_model_pretrained = keras.applications.ResNet50(include_top = False, #so that we can change input and output layer
                                            weights=weights_to_use, 
                                            input_tensor=input_resnet)
    my_model = Sequential()
    my_model.add(input_my_model)
    my_model.add(Conv3D(filters=16, kernel_size=(1,1,3), strides=(1,1,1), activation= 'relu'))
    my_model.add(Conv3D(filters=32, kernel_size=(1,1,3), strides=(1,1,1), activation= 'relu'))
    my_model.add(Conv3D(filters=64, kernel_size=(1,1,3), strides=(1,1,1), activation= 'relu'))
    my_model.add(Conv3D(filters=3, kernel_size=(1,1,3), strides=(1,1,1), activation= 'relu'))
    my_model.add(Conv3D(filters=1, kernel_size=(1,1,2), strides=(1,1,1), activation= 'relu'))
    my_model.add(res_model_pretrained)
    my_model.add(Flatten())
    my_model.add(layers.Dropout(dropout))
    my_model.add(layers.Dense(n_bins, activation='softmax'))
    my_model.summary()
    
    my_model.compile(loss='categorical_crossentropy',
                     optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                     metrics=['categorical_crossentropy'])
    
    return my_model
    



"My model creation"
