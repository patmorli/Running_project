#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:16:41 2023

@author: patrick
"""

from keras import layers
from keras.models import Sequential
from keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
import keras
from keras import backend as K
import tensorflow as tf

"""create a custom model for classification.
train X is the input data. layers_nodes is a list where the 
size represents the number of layers and the numbers of nodes in each layer"""
def lstm_model_class(input_my_model, layers_nodes, learning_rate):
    # model_specifics is a list where the size represents the number of
    # layers and the numbers the nodes in each layer
    model = Sequential()
    model.add(input_my_model)
    if len(layers_nodes) == 1:
        model.add(LSTM(layers_nodes[0], activation='relu', return_sequences=False))
    else: 
        model.add(LSTM(layers_nodes[0], activation='relu', return_sequences=True))
    
    
    for i in range(0,len(layers_nodes)-1):
        if i == len(layers_nodes)-2:
            model.add(LSTM(layers_nodes[i+1], activation='relu', return_sequences=False))
        else:
            model.add(LSTM(layers_nodes[i+1], activation='relu', return_sequences=True))
        
    
        
    model.add(layers.Dense(3, activation = 'softmax'))
    
    model.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics=['categorical_crossentropy'])
    print(model.summary())
    return model

def lstm_model_class_try(input_my_model, layers_nodes, learning_rate):
    model = Sequential()
    model.add(input_my_model)
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(layers.Dense(10, activation = 'softmax'))
    
    model.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics=['categorical_crossentropy'])
    print(model.summary())
    return model

def lstm_model_cont(input_my_model, layers_nodes, dropout):
    # model_specifics is a list where the size represents the number of
    # layers and the numbers the nodes in each layer
    model = Sequential()
    model.add(input_my_model)
    if len(layers_nodes) == 1:
        model.add(LSTM(layers_nodes[0], activation='relu', return_sequences=False))
    else: 
        model.add(LSTM(layers_nodes[0], activation='relu', return_sequences=True))
    
    
    for i in range(0,len(layers_nodes)-1):
        if i == len(layers_nodes)-2:
            model.add(LSTM(layers_nodes[i+1], activation='relu', return_sequences=False))
        else:
            model.add(LSTM(layers_nodes[i+1], activation='relu', return_sequences=True))
        
    
    model.add(layers.Dropout(dropout))    
    model.add(layers.Dense(1, activation = 'relu'))
    
    def root_mean_squared_error(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
    model.compile(loss=root_mean_squared_error,
                    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
                    metrics=root_mean_squared_error)
    print(model.summary())
    return model
    

"""Create the bidirectional lstm model. Needs X_train and 
y_train to understand the nature of the model input and output.
Input: X_train, ytrain 
Output: The prepared model"""
def bilstm_model_class(X_train, y_train):
    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional(
          keras.layers.LSTM(
              units=128,
              input_shape=[X_train.shape[1], X_train.shape[2]]
          )
        )
    )
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
  
    return model
