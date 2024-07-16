#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:31:23 2022

@author: patrickmayerhofer
script_running_classification_v5

loads the tfrecords, and uses them to do optimize a neural network

Based on:
https://www.kaggle.com/code/danmaccagnola/activity-recognition-data-w-tfrecords/notebook

and for model this could help. consider using a cnn-lstm model
https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
"""
import keras
import tensorflow as tf
import random
import functions_classification_general as fcg
import functions_my_model_resnet as fmm
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score



def script_running_classification_spectrogram_v1(subjects, dir_root, n_timesteps, model_name, 
                                                 n_features, examples_per_file, weights_to_use, 
                                                 val_split, epochs, batch_size, dropout,
                                                 early_stopping_patience, early_stopping_min_delta,
                                                 input_my_model, input_resnet, which_spectrograms,
                                                 resnet_trainable, n_bins, classification,flag_shuffle_files):
    import sys 
    sys.path.append('/Volumes/GoogleDrive/My Drive/Cycling Project/2021/Python/')
    
    
    
    """directories"""
    dir_data = dir_root + 'Data/'
    dir_prepared = dir_data + 'Prepared/'
    dir_tfr_spectrogram = dir_prepared + "tfrecords/" + which_spectrograms
    dir_results_weights = dir_data + 'Results/' + 'model_weights/'
    dir_results_history = dir_data + 'Results/' + 'model_history/'
    
    
    
    """get all ddata irectories"""
    filenames = list()
    for subject in subjects:
        sensor = "SENSOR" + "{:03d}".format(subject)
        dir_subject = dir_tfr_spectrogram + sensor + ".tfrecords"
        filenames.append(dir_subject)
    
    
    
    #shuffle subject list
    if flag_shuffle_files:
        random.shuffle(filenames)
    
    print("filenames:")
    print(filenames)
    
    
    """divide in train and test set directories"""
    val_filenames = filenames[0:int(len(filenames)*val_split)]
    train_filenames = filenames[int(len(filenames)*val_split):len(filenames)]
    
    print("val_filenames:")
    print(val_filenames)
    print("train_filenames:")
    print(train_filenames)
    
    """
    test_filenames = filenames[0:int(len(filenames)*test_split)]
    val_filenames = filenames[int(len(filenames)*test_split):int(len(filenames)*test_split)+int(len(filenames)*val_split)]
    train_filenames = filenames[int(len(filenames)*test_split)+int(len(filenames)*val_split):len(filenames)]
    """
    
    print(f"Train: {len(train_filenames)}")
    print(f"Validation: {len(val_filenames)}")
    #print(f"Test: {len(test_filenames)}")
    

    
    "callbacks"
    print('model checkpoint included')
    check_point = keras.callbacks.ModelCheckpoint(filepath= dir_results_weights + model_name + '.h5',
                                                 verbose = 1,
                                                 monitor="val_loss",
                                                 save_best_only=True,
                                                 mode="min", # if we save_best_only, we need to specify on what rule. Rule here is if val_loss is minimum, it owerwrites
                                                 save_weights_only = True,  # to only save weights, otherwise it will save whole model
                                                 )
    print('early stopping included') 
    
    # make sure to add this to the fit model again when uncommenting
    earlystopping = tf.keras.callbacks.EarlyStopping( 
                    monitor='val_loss',
                    min_delta=early_stopping_min_delta,
                    patience=early_stopping_patience,
                    verbose=1,
                    mode='auto',
                    restore_best_weights=True # Whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used.
                    )
    
    "Creating my_model"
    if classification:
        my_model = fmm.create_my_model_resnet50_class(input_my_model, input_resnet, weights_to_use, dropout, n_bins)
    
    if classification == 0:
        my_model = fmm.create_my_model_resnet50_cont(input_my_model, input_resnet, weights_to_use, dropout, n_bins)
        
    
    "editing my_model"
    my_model.layers[1].trainable = resnet_trainable# trainable weights of resnet50
    
    # check which parts overall are frozen
    for i, layer in enumerate(my_model.layers):
        print(i, layer.name, "-", layer.trainable)
     
        
    "if we want to change trainable in each individual layer in the resnet50 part"  
    """
    for layer in my_model.layers[1].layers[143:]:
        layer.trainable = True
        
    for i, layer in enumerate(my_model.layers[1].layers):
        print(i, layer.name, "-", layer.trainable)    
    """ 
    
    """
    for layer in my_model.layers:
        print(layer.output_shape)
    """
    
    
    "get data"
    if classification:
        train_dataset = fcg.get_dataset_bins(train_filenames, batch_size)
        val_dataset = fcg.get_dataset_bins_unshuffled(val_filenames, batch_size)
        #test_dataset = fcg.get_dataset(test_filenames, batch_size)
    
    if classification == 0:
        train_dataset = fcg.get_dataset_cont(train_filenames, batch_size)
        val_dataset = fcg.get_dataset_cont(val_filenames, batch_size)
    
    """
    if classification:
        for batch in tf.data.TFRecordDataset(val_filenames).map(fcg.parse_tfr_element_bins):
            tens = batch
            print(batch[1])
     """       
    
    
    "run optimization"
    
    history = my_model.fit(train_dataset, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=val_dataset,
                        callbacks=[check_point, earlystopping],
                        shuffle = False
                        )
    
        
    
    #print('NOT saving model again after training. Only during training.')
    my_model.save(dir_results_weights + model_name + '.h5')
    print("Saved model with the name: " + model_name)           
            
    
    
    
    """calculate accuracy in different ways"""
    # automatic
    evaluated_val_loss = my_model.evaluate(val_dataset)
    print('Val loss automatic: ' + str(evaluated_val_loss))
    evaluated_train_loss = my_model.evaluate(train_dataset)
    print('Train loss automatic: ' + str(evaluated_train_loss))
    #evaluated_test_accuracy = my_model.evaluate(test_dataset)
    #print('Test loss: ' + str(evaluated_test_accuracy))
    
    
    
    
    
    #manually
    steps_to_take = len(val_filenames)
    
   
    pred_list_val = []
    true_list_val = []
    
    for x, y in val_dataset.take(steps_to_take):
        
        
        pred_values_val = my_model.predict(x)
        
        pred_list_val = pred_list_val + list(pred_values_val)
        
        #pred_list = pred_list + list(pred)
        true_list_val = true_list_val + list(y.numpy())
    
    if classification:
        pred_list_val_argmax = np.argmax(pred_list_val, axis=1)
        true_list_val_argmax = np.argmax(true_list_val, axis=1)
        f1 = f1_score(true_list_val_argmax, pred_list_val_argmax)
        print('F1_score: ', str(f1))
        
        cm = confusion_matrix(y_true=true_list_val_argmax, y_pred=pred_list_val_argmax)
        print(cm)
        """
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        """
        
    # this needs to be doublechecked if ever used again
    if classification == 0:
        mse = tf.keras.losses.MeanSquaredError()
        mean_abs_error_val_function = mse(true_list_val, pred_list_val).numpy
        print('Val loss manually: ' + str(mean_abs_error_val_function))
    
    
        #calculate loss for mean of training data, calculate mean absolute error
        #vs true validation data to compare to performance of network
        steps_to_take = len(train_filenames)
        
       
        pred_list_train = []
        true_list_train = []
        
        
        for x, y in train_dataset.take(steps_to_take):
            
            pred_values_train = my_model.predict(x)
            
            pred_list_train = pred_list_train + list(pred_values_train)
            #pred_list = pred_list + list(pred)
            true_list_train = true_list_train + list(y.numpy())
        
        mean_true_list_train = np.full((len(true_list_train),1), np.mean(true_list_train))
        
        
        mean_abs_error_train_function_baseline = mse(true_list_val, mean_true_list_train).numpy
        print('Train mean vs true val (MSE): ' + str(mean_abs_error_train_function_baseline))
        
        #do the same for validation data
        mean_true_list_val = np.full((len(true_list_val),1), np.mean(true_list_val))
        mean_abs_error_val_function_baseline = mse(true_list_val, mean_true_list_val).numpy
        print('Val mean vs true val (MSE): ' + str(mean_abs_error_val_function_baseline))
    
    
    
    """Save some stuff"""
    my_variables = [history.history, my_model, evaluated_train_loss, evaluated_val_loss, val_filenames, subjects, true_list_val, pred_list_val,x,y]
    
    # save loss and val_loss as pkl
    with open(dir_results_history + model_name + '.pkl', 'wb') as file_pi:
        pickle.dump(my_variables, file_pi)
    
    return evaluated_train_loss, evaluated_val_loss

    
    
    
    
    
    
    
    #pred_values_test = my_model.predict(test_dataset)
    
    """
    steps_to_take = len(test_filenames)
    
    pred_values_list = []
    pred_list = []
    true_list = []
    
    for x, y in test_dataset.take(steps_to_take):
        
        pred_value = model.predict(x)
        pred_values_list = pred_values_list + list(pred_value)
    """
    
    """
    real_data_test = list(test_dataset.as_numpy_iterator())
    #true_values_test = real_data_test[0][1]
    
    
    true_values_temp = list()
    for i in range(len(real_data_test)):
        true_values_temp.append(real_data_test[i][1])
        
    #flatten list if not flat
    true_values_test = [item for sublist in true_values_temp for item in sublist]
    
    
    #manually calculated accuracy
    #pred_values_only_last = list(pred_values[:, len(pred_values[0,:,0])-1, 0])
    error = list()
    for i in range(0, len(pred_values_test)):
        error.append(abs(pred_values_test[i]-true_values_test[i]))
    
    mean_abs_error_test = sum(error)/len(error)
    """
    
    """Save some stuff"""
    """
    my_variables = [history.history, true_values_test, pred_values_test, evaluated_train_accuracy, evaluated_val_accuracy, evaluated_test_accuracy]
    # save loss and val_loss as pkl
    with open(dir_results_history + model_name + '.pkl', 'wb') as file_pi:
        pickle.dump(my_variables, file_pi)
    """
        
    """  
    # for loading
    with open(dir_results_history + model_name + '.pkl', 'rb') as file_pi:
         new_history = pickle.load(file_pi)   
   """ 
   
    
