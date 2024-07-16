#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:36:02 2022

@author: patrick
"""
import pickle
import functions_classification_general as fcg
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils.vis_utils import plot_model
from keras.models import Model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import pandas as pd
import functions_running_data_preparation as prep


model_name = '10k_1000_250_0_resnet50_12channels_bin_label_True_None_2023-06-02 17:56:00'
flag_plot_loss = 1
flag_plot_model = 0
flag_visualize_activations = 0
flag_plot_confusion_matrix = 1
flag_plot_datapoints = 1
classification = 1
subject = 100 # for activation visualization
batch_size = 32
n_bins = 2
bins_by_time_or_numbers = 0 # 1 = by time, 0 = by numbers (this one evenly distributes participants in bins)
plot_bins_flag = 1

"""directories"""
dir_root = '/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Running Plantiga Project/'
dir_data = dir_root + 'Data/'
dir_prepared = dir_data + 'Prepared/'
dir_tfr_spectrogram = dir_prepared + "tfrecords/"  + model_name +  ".pkl/Treadmill/"
dir_results_weights = dir_data + 'Results/' + 'model_weights/'
dir_results_history = dir_data + 'Results/' + 'model_history/'

"""load overview file"""
overview_file = pd.read_csv(dir_data + 'my_overview_file.csv')

"""Load results and model"""
my_model = keras.models.load_model(dir_results_weights + model_name + '.h5')
with open(dir_results_history + model_name + '.pkl', 'rb') as file_pi:
    [history, evaluated_train_loss, evaluated_val_loss, val_filenames, subjects, val_dataset_true, val_dataset_pred, val_dataset_seconds_true, val_dataset_seconds_pred] = pickle.load(file_pi)  
    
if classification:
    if flag_plot_datapoints:
        # change val_filenames properly (only needed if data coming from colab)
        if val_filenames[0][0:8] == '/content':
            for filename in range(len(val_filenames)):
                val_filenames[filename] = val_filenames[filename].replace('/content/drive', '/Volumes/GoogleDrive')
       
        #f1 = f1_score(true_list_val_argmax, pred_list_val_argmax)
        #prediction = my_model.predict(val_filenames)
        
        # check out where about the validation dataset lies on range of seconds
        """
        # get dataset with bins and predict and get true values
        steps_to_take = len(val_filenames)
        val_dataset = fcg.get_dataset_bins_unshuffled(val_filenames, batch_size)
        val_dataset_true, val_dataset_pred_, x = fcg.get_predictions_true_manually(val_dataset, my_model, steps_to_take)
        val_dataset_pred = np.argmax(val_dataset_pred_, axis=1)
        val_dataset_true = np.argmax(val_dataset_true, axis=1)
        
        
        
        # get same dataset but with seconds
        val_dataset_seconds = fcg.get_dataset_cont_unshuffled(val_filenames, batch_size)
        val_dataset_seconds_true, val_dataset_seconds_pred, x = fcg.get_predictions_true_manually(val_dataset_seconds, my_model, steps_to_take)
        val_dataset_seconds_pred = np.argmax(val_dataset_seconds_pred, axis=1)
        """
        # get argmax
        val_dataset_pred_argmax = np.argmax(val_dataset_pred, axis=1)
        val_dataset_true_argmax = np.argmax(val_dataset_true, axis=1)
        
        # plot
        bin_subject_ids, bin_times = prep.get_bins_info(subjects, overview_file, n_bins, bins_by_time_or_numbers, plot_bins_flag)
        
        for datapoint in range(len(val_dataset_seconds_true)):
            if val_dataset_pred_argmax[datapoint] == val_dataset_true_argmax[datapoint]:
                plt.plot(datapoint, val_dataset_seconds_true[datapoint], marker = 'x', linestyle = '', color = 'g')
            else:
                plt.plot(datapoint, val_dataset_seconds_true[datapoint], marker = 'x', linestyle = '', color = 'r')
        

    if flag_plot_confusion_matrix:
        cm = confusion_matrix(y_true=val_dataset_true_argmax, y_pred=val_dataset_pred_argmax)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        
    
    
    

if flag_plot_loss:
    plt.figure()
    train_loss = plt.plot(history['loss'], label = 'train_loss')
    val_loss = plt.plot(history['val_loss'], label = 'val_loss')
    plt.legend()
    


# currently saves a png in the current directory. But always overwirtes, so no problem right now. 
if flag_plot_model:
    plot_model(my_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    

if flag_visualize_activations:
    # first load example data
    sensor = "SENSOR" + "{:03d}".format(subject)
    filename = dir_tfr_spectrogram + sensor + ".tfrecords"
    """parse serialized data function"""
    
    dataset_small = fcg.get_dataset_small(filename)
    """
    for batch in tf.data.TFRecordDataset(filename).map(fcg.parse_tfr_element):
        print(batch)
        break
    """

    for sample in dataset_small.take(1):
      #print(sample[0].shape)
      #print(sample[1].shape)
      #print(sample)
      one_image = sample[0][:,:,:]
      break
   
    one_image_numpy = one_image.numpy()
    one_image_numpy = one_image_numpy.reshape([1,126,40,12])   
    
    
    # plot original image
    fig, axs = plt.subplots(4, 3, figsize=(12, 8))
    for i, ax in enumerate(axs.flat):
        ax.pcolormesh(one_image_numpy[0,:,:,i], shading='gouraud')
    
    
    # Create a model that returns the output of the convolutional layer - could make this more dynamic for all layers
    conv_model = Model(inputs=my_model.input, outputs=my_model.get_layer('conv3d').output)
    activations = conv_model.predict(one_image_numpy)
    
    # plot output from convolutional layer
    fig, axs = plt.subplots(3,1, figsize=(12, 8))
    for i, ax in enumerate(axs.flat):
        ax.pcolormesh(activations[0,:,:,i,0], shading='gouraud')
    
    
        

 