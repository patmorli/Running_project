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
- This is optimized for the treadmill data


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
time_steps = 8000 # the length of a window of the data
step = 1000 # how much the window will slide
nperseg= 250 # length of a window for the frequency analysis of the spectrogram, 500 for 251, 251; 1120 for 223, 223 but f and Sxx need adjustments
noverlap=0 # how much the window overlaps with the previous window. 462 for 251, 251; 1080 for 223, 223 but f and Sxx need adjustments
#num_samples = 12 # each window is a sample. will save multiple windows in a tfrecord
#tread_or_overground = 'tread'
save_flag = 1
plot_flag = 0 # careful, this plots 100s of windows if not setting a break point\
plot_bins_flag = 0
flag_all_speeds = 0 # 1 means that we put all speeds in series, so there will be only one spectrogram per person. --> requires time_steps to be at 25000
flag_spectrogram = 0 # 1 if we want to save spectrograms, 0 if we save raw data
flag_save_by_speeds = 1 # create a folder for each speed, so that we can do runner ID
n_bins = 2
bins_by_time_or_numbers = 0 # 1 = by time, 0 = by numbers (this one evenly distributes participants in bins)
flag_normalize_data = 0


subjects = [1,2,3,8,9,15,20,21,24,28,32,33,35,36,41,42,45,49,55,59,79,108,110,111,122,131,133] #,11,12,13,15,16,17,18,19,20,21,23,24,25,27,28,30,32,33,34,35,36,37,38,40,41,42,43,45,46,48,49,52,53,54,55,58,59,60,61,62,63,66,67,68,69,70,72,73,74,77,79,80,82,84,85,87,88,89,90,91,92,93,94,95,96,98,99,100,101,102,104,105,106,107,108,110,111,112,113,114,115,116,118,119,120,122,123,125,126,127,128,130,131,132,133,135,138,139,140,142,143,146,147,150,151,154,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,173,174,176,177,179,180,182,184,185,186,188]
tread_or_overground = 'Treadmill'
trials = [2] #1,2,3. all subjects have trial 1
speeds = [2.5,3.0,3.5]

# my helpful directories
dir_root = '/Users/patrick/Google Drive/My Drive/Running Plantiga Project/Data/'
dir_prepared = dir_root + 'Prepared/'
dir_csv = dir_prepared + 'csv/'
dir_tfr = dir_prepared + "tfrecords/"
#dir_save = dir_tfr + "5k_"+ str(nperseg) + '_' + str(noverlap) + "/"
dir_save = dir_tfr + '5k_normalized/'
dir_parquet = dir_prepared + "parquet/"
dir_data_overground = dir_csv + 'Overground/'
dir_data_treadmill = dir_csv + 'Treadmill/'
dir_data_tfr_overground = dir_save + 'Overground/'
dir_data_tfr_treadmill = dir_save + 'Treadmill/'

# get all directories 
all_files = np.sort(glob.glob(dir_csv + "*/*"))
#np.random.shuffle(files)

overview_file = pd.read_csv(dir_root + 'my_overview_file.csv')

# fix the time_steps if we are using all three speeds together
if flag_all_speeds:
    time_steps = 75000
    step = 75000
    
# creating empty spectrogram_image file
t_length = int((time_steps - nperseg) / (nperseg - noverlap) + 1)
f_length = int((nperseg / 2) + 1)

if flag_all_speeds:
    num_spectrograms_per_subject = 1
elif flag_all_speeds == 0 and flag_save_by_speeds == 0:
    num_spectrograms_per_subject = int((25000 - time_steps)/(time_steps - (time_steps-step)) + 1) * len(speeds) * len(trials)
else:
    num_spectrograms_per_subject = int((25000 - time_steps)/(time_steps - (time_steps-step)) + 1)

bin_subject_ids, bin_times = prep.get_bins_info(subjects, overview_file, n_bins, bins_by_time_or_numbers, plot_bins_flag)



for subject in subjects:
    ## create a spectrogram (12 layers because 12 features) for each window and save a tfr with all windows
    output_list = []
    i_spectrogram = 0 
    if flag_all_speeds == 0:
        spectrogram_images = np.empty((num_spectrograms_per_subject, f_length,t_length,12)) #first 12: 3 speeds x 4 windows per run --> needs to be always changed
        
    else:
        spectrogram_images = np.empty((1, f_length,t_length,12))
    sensor = "SENSOR" + "{:03d}".format(subject)
    dir_files = list()
    print('running: ' + str(subject))
    if tread_or_overground == 'Treadmill':
        for trial in trials:
            for speed in speeds:
                
                # e.g.: SENSOR001_2.5
                if trial == 1:
                    filename = sensor + '_' + str(speed) + '.csv'
                
                # e.g.: SENSOR001_2.5_2
                if trial == 2 or trial == 3:
                    filename = sensor + '_' + str(speed) + '_' + str(trial) + '.csv'
                
                dir_files.append(dir_data_treadmill + filename)  
    
    # put an if here which checks whether it should be individual speeds or all together 
    for dir_file_number in range(len(dir_files)):
        # load the file
        one_df_cur = pd.read_csv(dir_files[dir_file_number])
        
        # delete non-usable data
        one_df_cur = one_df_cur[one_df_cur['usabledata'] == True].reset_index(drop=True)
        
        if flag_all_speeds == 0:
            df_cur = one_df_cur
            
            if df_cur.speed[0] == 2.5:
                speed = 0
            elif df_cur.speed[0] == 3.0:
                speed = 1
            else:
                speed = 2
        else:
            speed = 7 # won't work with one_hot encoding. Deal with it when needed
            if dir_file_number == 0:
                df_cur = one_df_cur
            else:
                df_cur = df_cur.append(one_df_cur)
                df_cur = df_cur.reset_index(drop = True)
        
        
        # get spectrograms of df_cur
        # if doing 3 speeds individually we can do that before going to the next speed
        if flag_all_speeds == 0:
            if flag_spectrogram:
                if flag_save_by_speeds == 0:
                    spectrogram_images, score_10k, seconds_10k, subject_id, my_bin, i_spectrogram = \
                        prep.get_spectrograms(df_cur, subject, bin_subject_ids, dir_file_number, i_spectrogram, 
                                         spectrogram_images, time_steps, step, signal, data_sampling_fq, 
                                         nperseg, noverlap, plot_flag)
                else: 
                    # here we have to get the spectrogram, but only one, and then also save it as tfrecord here I believe
                    i_spectrogram = 0
                    spectrogram_images, score_10k, seconds_10k, subject_id, my_bin, i_spectrogram = \
                        prep.get_spectrograms(df_cur, subject, bin_subject_ids, dir_file_number, i_spectrogram, 
                                         spectrogram_images, time_steps, step, signal, data_sampling_fq, 
                                         nperseg, noverlap, plot_flag)
                        
                        
                    # save it right here:
                    if save_flag:
                        if not os.path.exists(dir_save + tread_or_overground + '/speed' + str(speed)):
                            os.makedirs(dir_save + tread_or_overground + '/speed' + str(speed))  # creating TFRecords output folder
                        score_10k_array = np.full(num_spectrograms_per_subject, score_10k)
                        seconds_10k_array = np.full(num_spectrograms_per_subject, seconds_10k)
                        subject_id_array = np.full(num_spectrograms_per_subject, subject)
                        bin_array = np.full(num_spectrograms_per_subject, my_bin)
                        speed_array = np.full(num_spectrograms_per_subject, speed) 
                        if trial == 1:
                            filedir = dir_save + tread_or_overground + '/speed' + str(speed) + '/' + sensor
                        else:
                            filedir = dir_save + tread_or_overground + '/speed' + str(speed) + '/' + sensor + '_' + str(trial)
                        
                        count = prep.write_spectrograms_to_tfr(spectrogram_images, score_10k_array, seconds_10k_array, filedir, subject_id_array, bin_array, speed_array)
                        
            else:
                if flag_save_by_speeds == 0:
                    "when we want to save the data raw, and not within spectrogram form"
                    output_list = prep.get_raw_data(df_cur, bin_subject_ids, subject, dir_file_number, time_steps, step, output_list, speed, plot_flag, flag_normalize_data)
                else:
                    output_list = []
                    output_list = prep.get_raw_data(df_cur, bin_subject_ids, subject, dir_file_number, time_steps, step, output_list, speed, plot_flag, flag_normalize_data)
                    output_list_df = pd.DataFrame(output_list)
                    
                    if save_flag:
                        if not os.path.exists(dir_save + tread_or_overground + '/speed' + str(speed)):
                            os.makedirs(dir_save + tread_or_overground + '/speed' + str(speed))  # creating TFRecords output folder
                        if trial == 1:    
                            filedir = dir_save + tread_or_overground + '/speed' + str(speed) + '/' + sensor
                        else:
                            filedir = dir_save + tread_or_overground + '/speed' + str(speed) + '/' + sensor + '_' + str(trial)
                        
                        with tf.io.TFRecordWriter(
                            filedir + ".tfrecords"
                        ) as writer:
                            for index, row_sample in output_list_df.iterrows():
                                example = prep.create_example(row_sample)
                                writer.write(example.SerializeToString())
    # create dataframe    
    if flag_spectrogram == 0:
        output_list_df = pd.DataFrame(output_list)
    """now figure out how to save it similarly to the spectrogram"""
        
    # if doing all 3 speeds together, we do it here, when all 3 speeds have been added to df_cur 
    if flag_all_speeds:
        prep.spectrogram_images, score_10k, seconds_10k, subject_id, my_bin, i_spectrogram = \
            prep.get_spectrograms(df_cur, subject, bin_subject_ids, dir_file_number, i_spectrogram,
                             spectrogram_images, time_steps, step, signal, data_sampling_fq, 
                             nperseg, noverlap, plot_flag)
    if flag_spectrogram:
        score_10k_array = np.full(num_spectrograms_per_subject, score_10k)
        seconds_10k_array = np.full(num_spectrograms_per_subject, seconds_10k)
        subject_id_array = np.full(num_spectrograms_per_subject, subject)
        bin_array = np.full(num_spectrograms_per_subject, my_bin)
        speed_array = np.full(num_spectrograms_per_subject, speed) # careful, speed will be right if we save it like this

    
    filedir = dir_save + tread_or_overground + '/' + sensor
    
    if save_flag and flag_save_by_speeds == 0:
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
    #--------------------------------  
    
    
    
    
                

    