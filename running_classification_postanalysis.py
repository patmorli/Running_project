#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:36:02 2022

@author: patrick
"""
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, mean_squared_error, accuracy_score
import pandas as pd
import functions_models as fm
import keras
import functions_classification as fc
import tensorflow as tf
import numpy as np


# change / with : 
model_name = '10k_1000_250_0_resnet50_12channels_bin_label_True_None_2023-06-02 17:56:00'
data_name = "10k_1000_250_0"
flag_plot_accuracy = 1
flag_plot_confusion_matrix = 1
flag_plot_bins = 1

"""for testing again"""
subjects = [35,36,42,45,49,55,79,108,110,111,122,131,133] # also not needed in theory, but it is an input into the function that I use
test_subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,19,20,21,23,24,25,27,28]
input_resnet = keras.Input(shape=(126,40,12)) # input after my own layers into the resnet
dropout = 0
final_layer_size = 2
final_activation = 'softmax'



"""directories"""
dir_root = "/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Running Plantiga Project/"
dir_data = dir_root + 'Data/'
dir_prepared = dir_data + 'Prepared/'
dir_results_trained = dir_data + 'Results/' + 'models_trained/'
dir_results_info = dir_data + 'Results/' + 'models_info/'



"""load overview file"""
overview_file = pd.read_csv(dir_data + 'my_overview_file.csv')


"""Load results and model"""
#my_model = keras.models.load_model(dir_results_trained + model_name + '.h5')
with open(dir_results_info + model_name + '.pkl', 'rb') as file_pi:
    [history, variable_one, variable_two, model_parameters] = pickle.load(file_pi)  
  
best_accuracy = max(history['accuracy'])
best_accuracy_epoch = history['accuracy'].index(best_accuracy)
best_val_accuracy = max(history['val_accuracy'])  
best_val_accuracy_epoch = history['val_accuracy'].index(best_val_accuracy)
total_epochs = len(history['accuracy'])
  


print('Used Subjects: ')
print(model_parameters["subjects"])
print('Test accuracy:')
print(accuracy_score(variable_one, variable_two))
#print('Used Test Subjects: ')
#print(model_parameters["test_subjects"])
print('total epochs:')
print(total_epochs)
print('Train Accuracy:')
print(best_accuracy)
print('val_accuracy and epoch')
print(best_val_accuracy)
print(best_val_accuracy_epoch)






if flag_plot_confusion_matrix:
    cm = confusion_matrix(y_true=variable_one, y_pred=variable_two, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

        
 

if flag_plot_accuracy:
    if "seconds_10k" in model_name:
        plt.figure()
        train_loss = plt.plot(history['loss'], label = 'train_loss')
        val_loss = plt.plot(history['val_loss'], label = 'val_loss')
        plt.legend()
    else:
        plt.figure()
        train_loss = plt.plot(history['accuracy'], label = 'train_accuracy')
        val_loss = plt.plot(history['val_accuracy'], label = 'val_accuracy')
        plt.legend()
  
if flag_plot_bins:
    dir_tfr = dir_prepared + "tfrecords/" + model_parameters['data_name'] + '/'
    reinitialize_epochs = 40
    flag_top_5_accuracy = 0
    learning_rate = 0.00001
    batch_size = 32
    threshold = 0.5
    filepath = dir_results_trained + model_name + '.h5'
    model = fm.resnet50_12channels_gpt(model_parameters['weights_to_use'], input_resnet, dropout, final_layer_size, final_activation, dir_results_trained)  
    model.load_weights(filepath) # to load the best weights again, no matter what. 
    train_filenames, val_filenames, test_filenames = fc.get_filenames_seconds(subjects, [1], dir_tfr, 0, test_subjects, 0.2)
    model, my_callbacks = fc.compile_model_callbacks_etc(filepath, model_parameters['output_variable'],  model_parameters['early_stopping_min_delta'], 
                                     model_parameters['early_stopping_patience'], reinitialize_epochs, model,flag_top_5_accuracy, learning_rate)
    
    variable_one, variable_two = fc.performance_test_dataset(model, test_filenames, batch_size, final_layer_size, 'resnet50_12channels', model_parameters['output_variable'], 10000, 0)
    
    test_dataset = fc.get_bin_and_seconds(test_filenames, batch_size)
    
    steps_to_take = len(test_filenames)
    
    bin_labels = []
    seconds_10k = []
    
    for x, y in test_dataset.take(steps_to_take):
        bin_label = tf.argmax(x,axis=1).numpy()
        
        bin_labels = bin_labels + list(bin_label)
        seconds_10k = seconds_10k + list(y)
        
    accuracy_score(bin_labels, variable_two)  
    
    # Convert true_labels and predicted_labels to numpy arrays and cast to float
    true_labels = np.array(variable_one).astype(float)
    predicted_labels = np.array(variable_two).astype(float)
    
    # Convert seconds_10k tensor to numpy array
    seconds = np.array(seconds_10k)
    
    # Create a list of x-coordinates
    x_coords = list(range(len(true_labels)))
    # Create a list of markers based on whether the true and predicted labels match
    markers = ['green' if true == pred else 'red' for true, pred in zip(true_labels, predicted_labels)]
    
    plt.figure()
    # Plot the data
    plt.scatter(x_coords, seconds, color=markers, label='True vs Predicted')
    plt.axhline(y=2745, color='b')
    
    # Add labels and legend
    plt.xlabel('Labels')
    plt.ylabel('Seconds')

    # Show the plot
    plt.show()


    
   
    
 
    


    
    
        

 