#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 18:59:59 2022

@author: patrickmayerhofer

Helpful resources:
https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c

required functions:
- functions_running_data_preparation

required files:
- prepared .csv files of participants
- overview file 


How it works:
- This file creates TFRecords. From CSV to Parquet to TFRecord
- Can create tfrecords with time series data
- Can create tfrecords of created spectrogram images
- Adds many variables to the TFRecord
- This is optimized for the overground data
"""



# my great libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys 
sys.path.append('/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Code/')
import functions_running_data_preparation as prep
import tensorflow as tf
import glob
np.random.seed(1111)
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt

from array import array
import keras


data_sampling_fq = 500

# changeable variables
y_mock_variable = ['left_ax']
time_steps = 8000
step = 7000
nperseg= 250 #500 for 251, 251; 1120 for 223, 223 but f and Sxx need adjustments
noverlap=0 #462 for 251, 251; 1080 for 223, 223 but f and Sxx need adjustments
#num_samples = 12 # each window is a sample. will save multiple windows in a tfrecord
#tread_or_overground = 'tread'
save_flag = 1
plot_flag = 0 # careful, this plots 100s of windows if not setting a break point\
plot_bins_flag = 1
flag_spectrogram = 1 # 1 if we want to save spectrograms, 0 if we save raw data
n_bins = 2
bins_by_time_or_numbers = 0 # 1 = by time, 0 = by numbers (this one evenly distributes participants in bins)
flag_normalize_data = 0


subjects = [1,2,3,8,9,20,21,24,28,32,33,35,36,42,45,49,55,79,108,110,111,122,131,133]
tread_or_overground = 'Overground'
trials = [3]

speed = 3 # for overground

# my helpful directories
dir_root = '/Users/patrick/Google Drive/My Drive/Running Plantiga Project/Data/'
dir_prepared = dir_root + 'Prepared/'
dir_csv = dir_prepared + 'csv/'
dir_tfr = dir_prepared + "tfrecords/"
dir_save = dir_tfr + "8k_"+ str(nperseg) + '_' + str(noverlap) + "/"
#dir_save = dir_tfr + '1k_no_spectrogram/'
dir_parquet = dir_prepared + "parquet/"
dir_data_overground = dir_csv + 'Overground/'
dir_data_treadmill = dir_csv + 'Treadmill/'
dir_data_tfr_overground = dir_save + 'Overground/'
dir_data_tfr_treadmill = dir_save + 'Treadmill/'

# get all directories 
all_files = np.sort(glob.glob(dir_csv + "*/*"))
#np.random.shuffle(files)

overview_file = pd.read_csv(dir_root + 'my_overview_file.csv')

    
# creating empty spectrogram_image file
t_length = int((time_steps - nperseg) / (nperseg - noverlap) + 1)
f_length = int((nperseg / 2) + 1)
    
bin_subject_ids, bin_times = prep.get_bins_info(subjects, overview_file, n_bins, bins_by_time_or_numbers, plot_bins_flag)



for subject in subjects:
    ## create a spectrogram (12 layers because 12 features) for each window and save a tfr with all windows
    output_list = []
    i_spectrogram = 0 
    
    
    sensor = "SENSOR" + "{:03d}".format(subject)
    dir_files = list()
    print('running: ' + str(subject))
    
    for trial in trials:
        # e.g.: SENSOR001
        if trial == 1:
            filename = sensor + '_run.csv'
        
        # e.g.: SENSOR001_2.5_2
        if trial == 2 or trial == 3:
            filename = sensor + '_'  + 'run_' + str(trial) + '.csv'
        
        dir_files.append(dir_data_overground + filename)  
    
    for dir_file_number in range(len(dir_files)):
        # load the file
        df_cur = pd.read_csv(dir_files[dir_file_number])
        
        # delete non-usable data
        df_cur = df_cur[df_cur['usabledata'] == True].reset_index(drop=True)
        
        
        if flag_spectrogram:
            num_spectrograms_per_subject = int((len(df_cur) - time_steps)/(time_steps - (time_steps-step)) + 1)
            spectrogram_images = np.empty((num_spectrograms_per_subject, f_length,t_length,12)) # first 1: 

            spectrogram_images, score_10k, seconds_10k, subject_id, my_bin, i_spectrogram = \
                prep.get_spectrograms(df_cur, subject, bin_subject_ids, dir_file_number, i_spectrogram, 
                                 spectrogram_images, time_steps, step, signal, data_sampling_fq, 
                                 nperseg, noverlap, plot_flag)
            
        else:
            "when we want to save the data raw, and not within spectrogram form"
            output_list = prep.get_raw_data(df_cur, bin_subject_ids, subject, dir_file_number, time_steps, step, output_list, speed, plot_flag, flag_normalize_data)
    
    # create dataframe    
    if flag_spectrogram == 0:
        output_list_df = pd.DataFrame(output_list)
        
        
    else:
        score_10k_array = np.full(num_spectrograms_per_subject, score_10k)
        seconds_10k_array = np.full(num_spectrograms_per_subject, seconds_10k)
        subject_id_array = np.full(num_spectrograms_per_subject, subject)
        bin_array = np.full(num_spectrograms_per_subject, my_bin)
        speed_array = np.full(num_spectrograms_per_subject, speed) # careful
        
    if trial == 1:    
        filedir = dir_save + tread_or_overground + '/' + sensor
    else:
        filedir = dir_save + tread_or_overground + '/' + sensor + '_' + str(trial)
   
    if save_flag:
       if not os.path.exists(dir_save + tread_or_overground):
           os.makedirs(dir_save + tread_or_overground)  # creating TFRecords output folder
       if flag_spectrogram: 
           # because here we are not saving it by speeds, I will just create a 0 speed
           count = prep.write_spectrograms_to_tfr(spectrogram_images, score_10k_array, seconds_10k_array, filedir, subject_id_array, bin_array, speed_array)
       else:
           with tf.io.TFRecordWriter(
               filedir + ".tfrecords"
           ) as writer:
               for index, row_sample in output_list_df.iterrows():
                   example = prep.create_example(row_sample)
                   writer.write(example.SerializeToString()) 
     