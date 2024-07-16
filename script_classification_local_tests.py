#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:48:58 2023

@author: patrick
"""

import tensorflow as tf
import numpy as np
import functions_classification as fc
import functions_models as fm
import keras
import pickle
from keras.utils.vis_utils import plot_model
from datetime import datetime
"""
def script_classification(data_name, subjects, val_split, test_split, flag_shuffle_train, flag_plot, flag_classification_style, 
                          epochs, dropout, batch_size, early_stopping_min_delta, early_stopping_patience,
                          dir_root, trials, window_length, kernel_size_convolution, model_to_use,
                          weights_to_use, input_model, input_resnet, resnet_trainable, reinitialize_epochs):
"""
       
if 1:  
    data_name = "15k_500_250_0"
    subjects = [1,2,3,8,9,20,21,24,28,32,33,35,36]
    window_length = 15000
    trials = [1]
    model_to_use = 'resnet50' #'resnet50_12channels_gpt' #'parallel_channels_LSTM, 'parallel_channels_conv1D', 'resnet50'

    flag_shuffle_train = 1
    flag_plot = 0
    flag_classification_style = 0 #speed: 1, subject id: 0



    val_split = 0.2
    test_split = 0.2

    epochs = 2
    dropout = 0.2
    batch_size = 32

    early_stopping_min_delta = 0
    early_stopping_patience = 5
    reinitialize_epochs = 40


    kernel_size_convolution = 500
    input_model = keras.Input(shape = (126,60,12,1)) #keras.Input(shape = (126,40,12,1))  #keras.Input(shape = (10000,12)) # input for first layer
    input_resnet = keras.Input(shape=(126,60,3)) # input after my own layers into the resnet
    resnet_trainable = True
    weights_to_use = None #'imagenet'

    dir_root = "/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Running Plantiga Project/"

    
    
    """Create an individual name ID"""
    name_ID = data_name + '_' + model_to_use + '_' + str(flag_classification_style) + '_'
    if model_to_use == 'parallel_channels_conv1D':
        name_ID = name_ID + str(kernel_size_convolution) + '_' + str(dropout) + '_'
    if model_to_use == 'resnet50' or model_to_use == 'resnet50_12channels':
        name_ID = name_ID + str(resnet_trainable) + '_' + str(weights_to_use) + '_'
        
    name_ID = name_ID + datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    
    
    if flag_classification_style == 1:
        final_layer_size = 4
    elif flag_classification_style == 0:
        final_layer_size = 188
        
    
    
    """directories"""
    dir_data = dir_root + 'Data/'
    dir_prepared = dir_data + 'Prepared/'
    dir_tfr = dir_prepared + "tfrecords/" + data_name + '/'
    dir_tfr_treadmill = dir_tfr + 'Treadmill/'
    dir_tfr_overground = dir_tfr + 'Overground/'
    dir_results_model = dir_data + 'Results/' + 'models_trained/'
    dir_results_info = dir_data + 'Results/' + 'models_info/'
    
    
    if flag_classification_style:
        train_filenames, val_filenames, test_filenames = fc.get_filenames_speed_classification(subjects, test_split, val_split, dir_tfr_treadmill, flag_shuffle_train)
    elif flag_classification_style == 0:
        train_filenames, val_filenames = fc.get_filenames_subjectid_classification(subjects, trials, dir_tfr, flag_shuffle_train)
        
            
        """
        #to doublecheck
        (tf.data.TFRecordDataset(train_filenames)
         .map(fc.parse_tfrecord)
         .map(fc.prepare_sample_multipleinputs)
        )
        """
    
    
    # plot some data
    if flag_plot:
        if flag_classification_style:
            tense = list()
            for batch in tf.data.TFRecordDataset(train_filenames).map(fc.parse_tfrecord_rnn):
                #print(batch)
                tense.append(batch)
                #break
                
            fc.plot_accelerations(tense)
        
                 
    
    """fun part"""
    final_activation = 'sigmoid'
    if final_layer_size > 1:
        final_activation = 'softmax'
        
    
    if model_to_use == 'parallel_channels_conv1D':
        model = fm.parallel_channels_conv1D_alt(final_layer_size, final_activation, dropout, window_length, kernel_size_convolution)
        
    elif model_to_use == 'parallel_channels_LSTM': 
       model = fm.parallel_channels_LSTM(final_layer_size, final_activation, dropout)
       
    elif model_to_use == 'resnet50':
        model = fm.resnet50(weights_to_use, input_model, input_resnet, dropout, final_layer_size, final_activation,resnet_trainable)
    
    elif model_to_use == 'resnet50_12channels_gpt':
        model = fm.resnet50_12channels_gpt(weights_to_use, input_resnet, dropout, final_layer_size, final_activation)     
        
            
    else:
        'Please specify the model!'
        
    plot_model(model, to_file = 'my_model.png', show_shapes = True)
    
    loss_to_use = tf.keras.losses.BinaryCrossentropy()
    if final_layer_size > 1:
        loss_to_use = 'categorical_crossentropy'
    
    model.compile(loss = loss_to_use, 
                  optimizer = "adam", 
                  metrics = ["accuracy", 
                             #tf.keras.metrics.AUC(curve = 'ROC'),
                             #tf.keras.metrics.AUC(curve = 'PR'),
                             #tf.keras.metrics.Precision(),
                             #tf.keras.metrics.Recall(),
                             #tf.keras.metrics.PrecisionAtRecall(0.8) 
                            ]) #tf.keras.metrics.AUC(from_logits=True)
    if flag_classification_style:
        if model_to_use == 'parallel_channels_LSTM' or model_to_use == 'parallel_channels_conv1D':
            train_dataset = fc.get_raw_dataset_multipleinputs_speed(train_filenames, batch_size, window_length)
            val_dataset = fc.get_raw_dataset_multipleinputs_speed(val_filenames, batch_size, window_length)
        elif model_to_use == 'resnet50' or model_to_use == 'resnet50_12channels_gpt':
            train_dataset = fc.get_spectrogram_dataset_speed(train_filenames, batch_size)
            val_dataset = fc.get_spectrogram_dataset_speed(val_filenames, batch_size)
        
            
    elif flag_classification_style == 0:
        if model_to_use == 'parallel_channels_LSTM' or model_to_use == 'parallel_channels_conv1D':
            train_dataset = fc.get_raw_dataset_multipleinputs_subjectid(train_filenames, batch_size, window_length)
            val_dataset = fc.get_raw_dataset_multipleinputs_subjectid(val_filenames, batch_size, window_length)
        elif model_to_use == 'resnet50' or model_to_use == 'resnet50_12channels_gpt':
            train_dataset = fc.get_spectrogram_dataset_subjectid(train_filenames, batch_size)
            val_dataset = fc.get_spectrogram_dataset_subjectid(val_filenames, batch_size)
            
    
    
    "callbacks"
    
    print('model checkpoint included')
    filepath = dir_results_model + name_ID + '.h5'
    check_point = keras.callbacks.ModelCheckpoint(filepath,
                                                 verbose = 1,
                                                 monitor="val_accuracy",
                                                 save_best_only=True,
                                                 mode='auto', # if we save_best_only, we need to specify on what rule. Rule here is if val_loss is minimum, it owerwrites
                                                 #save_weights_only = True,  # to only save weights, otherwise it will save whole model
                                                 )
    
    
    
    # make sure to add this to the fit model again when uncommenting
    print('early stopping included') 
    earlystopping = tf.keras.callbacks.EarlyStopping( 
                    monitor='val_accuracy',
                    min_delta=early_stopping_min_delta,
                    patience=early_stopping_patience,
                    verbose=1,
                    mode='auto',
                    restore_best_weights=True # Whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used.
                    )
    
    reinitialize_callback = fc.ReinitializeWeightsCallback(reinitialize_epochs=reinitialize_epochs)


    
    #steps_per_epoch = steps_per_epoch
    
    history = model.fit(train_dataset,
              validation_data = val_dataset, 
              callbacks=[earlystopping, check_point, reinitialize_callback],
              #steps_per_epoch = steps_per_epoch,
              #validation_steps = validation_steps, 
              epochs = epochs
             )
    
    model.load_weights(filepath) # to load the best weights again, no matter what. 
    
    if flag_classification_style == 1:
        true_labels, pred_labels = fc.performance_test_dataset(model, test_filenames, batch_size, final_layer_size, model_to_use, flag_classification_style, window_length)
        
    else:
        true_labels, pred_labels = fc.performance_test_dataset(model, val_filenames, batch_size, final_layer_size, model_to_use, flag_classification_style, window_length)
    
    """Save some stuff"""
    model_parameters = {
        "subjects": subjects,
        "data_name": data_name,
        "model_to_use": model_to_use,
        "flag_classification_style": flag_classification_style,
        "epochs": epochs,
        "dropout": dropout,
        "early_stopping_min_delta": early_stopping_min_delta,
        "early_stopping_patience": early_stopping_patience,
        "kernel_size_convolution": kernel_size_convolution,
        "resnet_trainable": resnet_trainable,
        "weights_to_use": weights_to_use}
    
    my_variables = [history.history, true_labels, pred_labels, model_parameters]
    
    
    # save loss and val_loss as pkl
    with open(dir_results_info + name_ID + '.pkl', 'wb') as file_pi:
        pickle.dump(my_variables, file_pi)
    