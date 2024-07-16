#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:40:04 2022

@author: patrick
"""

import script_running_classification_spectrogram_v2 as script
import keras

"""some variables"""
"""
n_timesteps = 2000 # predefined, when we saved tfrecords in "create_TFRecords_spectrogram2"
n_features = 12 # same here. 3 accelerations, 3 angular velocities, 2 feet
examples_per_file = 12 #4 examples per speed, total number of examples in one tfrecord
"""

epochs = 100
subjects = [1,2,3,4,5,6,7,8,9,10]
val_split = 0.2
batch_size = 32
model_to_use = 'lstm_model_class' # resnet50_class_more_conv_layers, or resnet50_class, 'lstm_model_class'
weights_to_use = "imagenet"
dropout = 0
early_stopping_patience = 10
early_stopping_min_delta = 0
input_my_model = keras.Input(shape = (5000,1)) #keras.Input(shape = (126,40,12,1))  #keras.Input(shape = (10000,12)) # input for first layer
input_resnet = keras.Input(shape=(126,40,3)) # input after my own layers into the resnet
resnet_trainable = True
data_name = '5k_no_spectrogram'
model_name = 'local_speed'
which_spectrograms = data_name + '/Treadmill/'
classification = 1 # 0 if continuous
n_bins = 3
flag_shuffle_files = 0
flag_subject_id_classification = 0
flag_speed_classification = 1

test_speed = 3
layers_nodes = [16]
learning_rate = 2e-2


dir_root = '/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Running Plantiga Project/'

evaluated_train_loss, evaluated_val_loss = \
        script.script_running_classification_spectrogram_v2(subjects, dir_root, model_name,  weights_to_use, 
                                                 val_split, epochs, batch_size, dropout,
                                                 early_stopping_patience, early_stopping_min_delta,
                                                 input_my_model, input_resnet, which_spectrograms,
                                                 resnet_trainable, n_bins, classification,flag_shuffle_files,
                                                 model_to_use, layers_nodes, flag_subject_id_classification, test_speed,
                                                 learning_rate,flag_speed_classification)



