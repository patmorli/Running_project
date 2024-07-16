#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 18:59:59 2022

@author: patrickmayerhofer

create_TFRecords_spectrogram_decibel

This file creates TFRecords. From CSV to Parquet to TFRecord

Based on:
https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
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
time_steps = 25000
step = 5000
nperseg= 500 #500 for 251, 251; 1120 for 223, 223 but f and Sxx need adjustments
noverlap=250 #462 for 251, 251; 1080 for 223, 223 but f and Sxx need adjustments
#num_samples = 12 # each window is a sample. will save multiple windows in a tfrecord
#tread_or_overground = 'tread'
save_flag = 1
plot_flag = 0 # careful, this plots 100s of windows if not setting a break point\
plot_bins_flag = 0
flag_all_speeds = 0 # 1 means that we put all speeds in series, so there will be only one spectrogram per person. --> requires time_steps to be at 25000
n_bins = 2
bins_by_time_or_numbers = 1 # 1 = by time, 0 = by numbers (this one evenly distributes participants in bins)


subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,23,24,25,27,28,30,32,33,34,35,36,37,38,40,41,42,43,45,46,48,49,52,53,54,55,58,59,60,61,62,63,66,67,68,69,70,72,73,74,77,79,80,82,84,85,87,88,89,90,91,92,93,94,95,96,98,99,100,101,102,104,105,106,107,108,110,111,112,113,114,115,116,118,119,120,122,123,125,126,127,128,130,131,132,133,135,138,139,140,142,143,146,147,150,151,154,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,173,174,176,177,179,180,182,184,185,186,188] 
tread_or_overground = 'Treadmill'
trials = [1]
speeds = [2.5,3.0,3.5]

# my helpful directories
dir_root = '/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Data/'
dir_prepared = dir_root + 'Prepared/'
dir_csv = dir_prepared + 'csv/'
dir_tfr = dir_prepared + "tfrecords/"
dir_spectrogram = dir_tfr + "25k_"+ str(nperseg)+ '_' + str(noverlap) + "_bins2_parallelspeed/"
dir_parquet = dir_prepared + "parquet/"
dir_data_overground = dir_csv + 'Overground/'
dir_data_treadmill = dir_csv + 'Treadmill/'
dir_data_tfr_overground = dir_spectrogram + 'Overground/'
dir_data_tfr_treadmill = dir_spectrogram + 'Treadmill/'

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
else:
    num_spectrograms_per_subject = int((25000 - time_steps)/(time_steps - step) + 1) * len(speeds) * len(trials)


bin_subject_ids, bin_times = prep.get_bins_info(subjects, overview_file, n_bins, bins_by_time_or_numbers, plot_bins_flag)



for subject in subjects:
    ## create a spectrogram (12 layers because 12 features) for each window and save a tfr with all windows
    i_spectrogram = 0 
    if flag_all_speeds == 0:
        spectrogram_images = np.empty((num_spectrograms_per_subject, f_length,t_length,12)) #first 12: 3 speeds x 4 windows per run --> needs to be always changed
    else:
        spectrogram_images = np.empty((1, f_length,t_length,12))
    sensor = "SENSOR" + "{:03d}".format(subject)
    output_list = []
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
        else:
            if dir_file_number == 0:
                df_cur = one_df_cur
            else:
                df_cur = df_cur.append(one_df_cur)
                df_cur = df_cur.reset_index(drop = True)
        
        # get spectrograms of df_cur
        # if doing 3 speeds individually we can do that before going to the next speed
        if flag_all_speeds == 0:
            spectrogram_images, score_10k, seconds_10k, subject_id, my_bin, filename_and_id, i_spectrogram = \
                prep.get_spectrograms(df_cur, subject, bin_subject_ids, dir_file_number, i_spectrogram, 
                                 spectrogram_images, time_steps, step, signal, data_sampling_fq, 
                                 nperseg, noverlap, plot_flag)
        
    # if doing all 3 speeds together, we do it here, when all 3 speeds have been added to df_cur 
    if flag_all_speeds:
        prep.spectrogram_images, score_10k, seconds_10k, subject_id, my_bin, filename_and_id, i_spectrogram = \
            prep.get_spectrograms(df_cur, subject, bin_subject_ids, dir_file_number, i_spectrogram,
                             spectrogram_images, time_steps, step, signal, data_sampling_fq, 
                             nperseg, noverlap, plot_flag)
   
    score_10k_array = np.full(num_spectrograms_per_subject, score_10k)
    seconds_10k_array = np.full(num_spectrograms_per_subject, seconds_10k)
    subject_id_array = np.full(num_spectrograms_per_subject, subject)
    bin_array = np.full(num_spectrograms_per_subject, my_bin)

    
    filedir = dir_spectrogram + tread_or_overground + '/' + sensor
    
    if save_flag:
        count = prep.write_spectrograms_to_tfr(spectrogram_images, score_10k_array, seconds_10k_array, filedir, subject_id_array, bin_array)
    #--------------------------------  
    
    
    
    
                

    