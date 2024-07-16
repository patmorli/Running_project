#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:39:13 2023

@author: patrick
"""
import script_classification
import keras

data_name = "10k_1000_250_0"
subjects = [30,32,33,34,35,36,37,38,40,42,43,45,46,48,49,52,53,54,55,58,60,61,62,63,66,67,68,69,70,72,73,74,77,79,80,82,84,85,87,88,89,90,91,92,93,94,95,96,98,99,100,101,102,104,105,106,107,108,110,111,112,113,114,115,116,118,119,120,122,123,126,127,128,130,131,132,133,135,138,139,142,143,146,147,150,151,154,156,157,158,159,160,161,162,163]
window_length = 10000
trials = [1]
model_to_use = 'resnet50_12channels' #'resnet50_12channels_gpt' #'parallel_channels_LSTM, 'parallel_channels_conv1D', 'resnet50', 'resnet50_12channels'
# for performance testing, these need to be different from the subjects
test_subjects = [] #[1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,19,20,21,23,24,25,27,28] # if empty it will not use test subjects. For subject ID testing: If it is filled, it will use the overground data of these specfific subjects for test only. 
learning_rate = [] #[] is default of 0.001

flag_shuffle_train = 0 # maybe this can go, if we are shuffling subjects beforehand anyways
flag_shuffle_subjects = 0
flag_plot = 1
flag_top_5_accuracy = 0
output_variable = "bin_label" #"speed", "subject_id", "seconds_10k", "bin_label"
percentage_of_data_to_be_used = 1 # number between 0 and 1. 1 is all data, everything else takes the percentage 


val_split = 0.2
test_split = 0.2

epochs = 10
dropout = 0.2
batch_size = 32

early_stopping_min_delta = 0
early_stopping_patience = 5
reinitialize_epochs = 40


kernel_size_convolution = 500
input_model = keras.Input(shape = (100,60,12,1)) #keras.Input(shape = (126,40,12,1))  #keras.Input(shape = (10000,12)) # input for first layer
input_resnet = keras.Input(shape=(126,40,12)) # input after my own layers into the resnet
resnet_trainable = True
weights_to_use = None #None #'imagenet'

dir_root = "/Users/patmorli/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@weartechlabs.com/My Drive/Running Plantiga Project - Backup From Locomotion/"

name_id = script_classification.script_classification(data_name, subjects, val_split, test_split, flag_shuffle_train, flag_plot, output_variable, 
                          epochs, dropout, batch_size, early_stopping_min_delta, early_stopping_patience,
                          dir_root, trials, window_length, kernel_size_convolution, model_to_use,
                          weights_to_use, input_model, input_resnet, resnet_trainable, reinitialize_epochs, flag_shuffle_subjects,
                          flag_top_5_accuracy, test_subjects, learning_rate,percentage_of_data_to_be_used)