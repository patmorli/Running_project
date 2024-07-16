#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:48:58 2023

@author: patrick
"""

import tensorflow as tf
import functions_classification as fc
import functions_models as fm
import pickle
from datetime import datetime
import random
import pandas as pd

def script_classification(data_name, subjects, val_split, test_split, flag_shuffle_train, flag_plot, output_variable, 
                          epochs, dropout, batch_size, early_stopping_min_delta, early_stopping_patience,
                          dir_root, trials, window_length, kernel_size_convolution, model_to_use,
                          weights_to_use, input_model, input_resnet, resnet_trainable, reinitialize_epochs, flag_shuffle_subjects,
                          flag_top_5_accuracy, test_subjects, learning_rate, percentage_of_data_to_be_used):

    """    
if 1:  
    data_name = "15k_500_250_0"
    subjects = [1,2,3,8,9,20,21,24,28,32,33,35,36]
    window_length = 15000
    trials = [1,2,3]
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
    reinitialize_epochs = 10

    kernel_size_convolution = 500
    input_model = keras.Input(shape = (126,60,12,1)) #keras.Input(shape = (126,40,12,1))  #keras.Input(shape = (10000,12)) # input for first layer
    input_resnet = keras.Input(shape=(126,60,3)) # input after my own layers into the resnet
    resnet_trainable = True
    weights_to_use = None #'imagenet'

    dir_root = "/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Running Plantiga Project/"
    """
    
    
    
    # first, shuffle subjects if we want too
    if flag_shuffle_subjects:
        random.shuffle(subjects)
        
    
    """Create an individual name ID"""
    name_ID = data_name + '_' + model_to_use + '_' + output_variable + '_'
    if model_to_use == 'parallel_channels_conv1D':
        name_ID = name_ID + str(kernel_size_convolution) + '_' + str(dropout) + '_'
    if model_to_use == 'resnet50' or model_to_use == 'resnet50_12channels':
        name_ID = name_ID + str(resnet_trainable) + '_' + str(weights_to_use) + '_'
        
    name_ID = name_ID + datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    
    if weights_to_use != (None and 'imagenet'):
        name_ID = name_ID + '_transfer'
        print('Using transfer learning.')
    
    
    if output_variable == "speed":
        final_layer_size = 4
        print("output_variable = speed")
    elif output_variable == "subject_id":
        final_layer_size = 188
        print("output_variable = subject_id")
    elif output_variable == "seconds_10k":
        final_layer_size = 1
        print("output_variable = seconds_10k")
    elif output_variable == 'bin_label':
        final_layer_size = 2
        print("output_variable = bin_label --> currently with 2 bins")
        
    else:
        print("Specify output_variable correctly.")
        
    
    
    """directories"""
    dir_data = dir_root + 'Data/'
    dir_prepared = dir_data + 'Prepared/'
    dir_tfr = dir_prepared + "tfrecords/" + data_name + '/'
    dir_tfr_treadmill = dir_tfr + 'Treadmill/'
    dir_tfr_overground = dir_tfr + 'Overground/'
    dir_results_model = dir_data + 'Results/' + 'models_trained/'
    dir_results_info = dir_data + 'Results/' + 'models_info/'
    
    
    if output_variable == 'speed':
        train_filenames, val_filenames, test_filenames = fc.get_filenames_speed_classification(subjects, test_split, val_split, dir_tfr_treadmill, flag_shuffle_train)
        
    elif output_variable =='subject_id':
        train_filenames, val_filenames, test_filenames = fc.get_filenames_subjectid_classification(subjects, trials, dir_tfr, flag_shuffle_train, test_subjects)
        
    elif output_variable == "seconds_10k" or output_variable == "bin_label":
        train_filenames, val_filenames, test_filenames = fc.get_filenames_seconds(subjects, trials, dir_tfr, flag_shuffle_train, test_subjects, val_split)
    else:
        print('output_variable wrong')
        
            
        """
        #to doublecheck
        (tf.data.TFRecordDataset(train_filenames)
         .map(fc.parse_tfrecord)
         .map(fc.prepare_sample_multipleinputs)
        )
        """
    
    
    # plot some data
    # this might not work anymore, not sure if it was written for speed or subject_id
    if flag_plot:
        if output_variable == "speed": 
            tense = list()
            for batch in tf.data.TFRecordDataset(train_filenames).map(fc.parse_tfrecord_rnn):
                #print(batch)
                tense.append(batch)
                #break
                
            fc.plot_accelerations(tense)
        
                 
    
    """fun part"""
    final_activation = 'sigmoid'
    if final_layer_size > 1 and (output_variable == "speed" or output_variable == "subject_id" or output_variable == "bin_label"):
        final_activation = 'softmax'
    elif output_variable == "seconds_10k":
        final_activation = 'relu'  
        print("remember to also try linear as final_activation. currently using relu.")
        final_layer_size = 1
    else:
        print("final_activation could not properly be chosen")
    
    if model_to_use == 'parallel_channels_conv1D':
        model = fm.parallel_channels_conv1D_alt(final_layer_size, final_activation, dropout, window_length, kernel_size_convolution)
        
    elif model_to_use == 'parallel_channels_LSTM': 
       model = fm.parallel_channels_LSTM(final_layer_size, final_activation, dropout)
       
    elif model_to_use == 'resnet50':
        model = fm.resnet50(weights_to_use, input_model, input_resnet, dropout, final_layer_size, final_activation,resnet_trainable)
    
    elif model_to_use == 'resnet50_12channels':
        model = fm.resnet50_12channels_gpt(weights_to_use, input_resnet, dropout, final_layer_size, final_activation, dir_results_model)     
        
            
    else:
        print('Please specify the model!')
        
    #plot_model(model, to_file = 'my_model.png', show_shapes = True)
    
    
    
    "get my datasets"
    #train_dataset = fc.get_spectrogram_dataset(train_filenames, batch_size, output_variable, percentage_of_data_to_be_used)
    #val_dataset = fc.get_spectrogram_dataset(val_filenames, batch_size, output_variable, percentage_of_data_to_be_used)
    
    #to test new data fetching
    train_dataset = fc.get_partial_dataset(train_filenames, batch_size, output_variable, percentage_of_data_to_be_used)
    val_dataset = fc.get_partial_dataset(val_filenames, batch_size, output_variable, percentage_of_data_to_be_used)
    
    
    # to test original function that did not include data reduction
    #train_dataset =  fc.get_spectrogram_dataset_back_to_the_roots(train_filenames, batch_size, output_variable)
    #val_dataset =  fc.get_spectrogram_dataset_back_to_the_roots(val_filenames, batch_size, output_variable)
   
    
   
    first_element_dataset = train_dataset.take(1)

    # Iterate over the first element dataset and print its contents
    for element in first_element_dataset:
        print(element[0].shape)
        print(element[1].shape)
    
   
    
    # when script is working, this could be deleted, unless we want to think about using lstm or parallel conv at some point again
    """
    if output_variable == "speed":
        if model_to_use == 'parallel_channels_LSTM' or model_to_use == 'parallel_channels_conv1D':
            train_dataset = fc.get_raw_dataset_multipleinputs_speed(train_filenames, batch_size, window_length)
            val_dataset = fc.get_raw_dataset_multipleinputs_speed(val_filenames, batch_size, window_length)
        elif model_to_use == 'resnet50' or model_to_use == 'resnet50_12channels':
            train_dataset = fc.get_spectrogram_dataset_speed(train_filenames, batch_size)
            val_dataset = fc.get_spectrogram_dataset_speed(val_filenames, batch_size)
        
            
    elif output_variable == "subject_id":
        if model_to_use == 'parallel_channels_LSTM' or model_to_use == 'parallel_channels_conv1D':
            train_dataset = fc.get_raw_dataset_multipleinputs_subjectid(train_filenames, batch_size, window_length)
            val_dataset = fc.get_raw_dataset_multipleinputs_subjectid(val_filenames, batch_size, window_length)
        elif model_to_use == 'resnet50' or model_to_use == 'resnet50_12channels':
            train_dataset = fc.get_spectrogram_dataset_subjectid(train_filenames, batch_size)
            val_dataset = fc.get_spectrogram_dataset_subjectid(val_filenames, batch_size)
      """      
    
    
    filepath = dir_results_model + name_ID + '.h5'
    model, my_callbacks = fc.compile_model_callbacks_etc(filepath, output_variable, early_stopping_min_delta, 
                                    early_stopping_patience, reinitialize_epochs, model,flag_top_5_accuracy, learning_rate)


    
    #steps_per_epoch = steps_per_epoch
    
    history = model.fit(train_dataset,
              validation_data = val_dataset, 
              callbacks=my_callbacks,
              #steps_per_epoch = steps_per_epoch,
              #validation_steps = validation_steps, 
              epochs = epochs
             )
    
    model.load_weights(filepath) # to load the best weights again, no matter what. 

    if output_variable == "speed":
        # these are different between classification and regression, that's why they are called variable_one and variable_two
        variable_one, variable_two = fc.performance_test_dataset(model, test_filenames, batch_size, final_layer_size, model_to_use, output_variable, window_length)
        
    else:
        """for comparison with mean of training seconds --> no if statement, because input is needed for performance_test_dataset function"""
        seconds_file = pd.read_csv(dir_data + 'Subject_seconds_scores.csv')
        seconds_participants_train = seconds_file[seconds_file["ID"].astype(str).isin(map(str, subjects))]   
        mean_seconds_training  = int(seconds_participants_train["seconds_10k"].mean())
        if test_subjects == []:
            print("Testing performance on val set.")
            variable_one, variable_two = fc.performance_test_dataset(model, val_filenames, batch_size, final_layer_size, model_to_use, output_variable, window_length, mean_seconds_training, percentage_of_data_to_be_used = 1)
        else:
            print("Testing performance on test set.")
            variable_one, variable_two = fc.performance_test_dataset(model, test_filenames, batch_size, final_layer_size, model_to_use, output_variable, window_length, mean_seconds_training, percentage_of_data_to_be_used = 1)
            
    
    """Save some stuff"""
    model_parameters = {
        "subjects": subjects,
        "test_subjects": test_subjects,
        "data_name": data_name,
        "model_to_use": model_to_use,
        "output_variable": output_variable,
        "epochs": epochs,
        "dropout": dropout,
        "early_stopping_min_delta": early_stopping_min_delta,
        "early_stopping_patience": early_stopping_patience,
        "kernel_size_convolution": kernel_size_convolution,
        "resnet_trainable": resnet_trainable,
        "weights_to_use": weights_to_use}
    
    my_variables = [history.history, variable_one, variable_two, model_parameters]
    
    
    # save loss and val_loss as pkl
    with open(dir_results_info + name_ID + '.pkl', 'wb') as file_pi:
        pickle.dump(my_variables, file_pi)
        
    print("saved model as as: " + name_ID)
    print("subjects: " + str(subjects)) 
    print("test subjects: " + str(test_subjects))
    return name_ID
    