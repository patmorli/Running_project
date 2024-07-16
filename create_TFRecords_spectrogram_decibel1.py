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
import librosa
import librosa.display

data_sampling_fq = 500

# changeable variables
y_mock_variable = ['left_ax']
time_steps = 10000
step = 5000
nperseg= 100#500 for 251, 251; 1120 for 223, 223 but f and Sxx need adjustments
noverlap=0 #462 for 251, 251; 1080 for 223, 223 but f and Sxx need adjustments
num_samples = 12 # each window is a sample. will save multiple windows in a tfrecord
#tread_or_overground = 'tread'
save_flag = 1
plot_flag = 0 # careful, this plots 100s of windows if not setting a break point

subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,23,24,25,27,28,30,32,33,34,35,36,37,38,40,41,42,43,45,46,48,49,52,53,54,55,58,59,60,61,62,63,66,67,68,69,70,72,73,74,77,79,80,82,84,85,87,88,89,90,91,92,93,94,95,96,98,99,100,101,102,104,105,106,107,108,110,111,112,113,114,115,116,118,119,120,122,123,125,126,127,128,130,131,132,133,135,138,139,140,142,143,146,147,150,151,154,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,173,174,176,177,179,180,182,184,185,186,188] #range(1,2)
tread_or_overground = 'Treadmill'
trials = [1]
speeds = [2.5,3.0,3.5]

# my helpful directories
dir_root = '/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Data/'
dir_prepared = dir_root + 'Prepared/'
dir_csv = dir_prepared + 'csv/'
dir_tfr = dir_prepared + "tfrecords/"
dir_spectrogram = dir_tfr + "windows_10000_spectrogram_log_nperseg100/"
dir_parquet = dir_prepared + "parquet/"
dir_data_overground = dir_csv + 'Overground/'
dir_data_treadmill = dir_csv + 'Treadmill/'
dir_data_tfr_overground = dir_spectrogram + 'Overground/'
dir_data_tfr_treadmill = dir_spectrogram + 'Treadmill/'

# get all directories 
all_files = np.sort(glob.glob(dir_csv + "*/*"))
#np.random.shuffle(files)

overview_file = pd.read_csv(dir_root + 'my_overview_file.csv')

# creating empty spectrogram_image file
t_length = int((time_steps - nperseg) / (nperseg - noverlap) + 1)
f_length = int((nperseg / 2) + 1)



num_spectrograms_per_subject = int((25000 - time_steps)/(time_steps - step) + 1) * len(speeds) * len(trials)


for subject in subjects:
    ## create a spectrogram (12 layers because 12 features) for each window and save a tfr with all windows
    i_spectrogram = 0 
    spectrogram_images = np.empty((12, f_length,t_length,12)) #first 12: 3 speeds x 4 windows per run --> needs to be always changed
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
                
    for dir_file in dir_files:
        # load the file
        df_cur = pd.read_csv(dir_file)
        
        # delete non-usable data
        df_cur = df_cur[df_cur['usabledata'] == True]
        
        # create output_list with moving windows
        score_10k = int(df_cur.score_10k.iloc[0])
        seconds_10k = int(df_cur.seconds_10k.iloc[0])
        subject_id = int(df_cur.subject_id.iloc[0])
        tread_or_overground_bool = int(df_cur.iloc[0,17])
        if df_cur.trial_id.iloc[0] == 1:
            filename_and_id = 'SENSOR' + str(subject) + '_' + str(df_cur.speed.iloc[0])
        if df_cur.trial_id.iloc[0] == 2:
            filename_and_id = 'SENSOR' + str(subject) + '_' + str(df_cur.speed.iloc[0]) + '_2'
        if df_cur.trial_id.iloc[0] ==3:
            filename_and_id = 'SENSOR' + str(subject) + '_' + str(df_cur.speed.iloc[0]) + '_3'
            
                
                
        # creates windows and saves in a dataframe
        file_id = 0
        for i in range(0, len(df_cur)-time_steps + 1, step):
            v = df_cur.iloc[i:(i + time_steps)]
            df_acc_and_angvel = v.iloc[:, 3:15]
            list_samples_cur = df_acc_and_angvel.to_dict('records')
            
            
            for feature in range(len(df_acc_and_angvel.iloc[0])):
                # calculate spectrogram
                f, t, Sxx = signal.spectrogram(df_acc_and_angvel.iloc[:,feature], data_sampling_fq, nperseg=nperseg, noverlap=noverlap)
                # convert power to decibel 
                Sxx_to_db1 = librosa.power_to_db(Sxx, ref=np.max)
                #f = f[0:223]
                #Sxx = Sxx[0:223,:]
                
                spectrogram_images[i_spectrogram,:,:,feature] = Sxx_to_db1
                
                if plot_flag:
                    plt.figure()
                    plt.pcolormesh(t, f, Sxx_to_db1, shading='gouraud')
                    plt.ylabel('Frequency [db]')
                    plt.xlabel('Time [sec]')
                    plt.title(df_acc_and_angvel.columns[feature] + str(df_cur['speed'].iloc[0]))
            i_spectrogram = i_spectrogram + 1   
    
    score_10k_array = np.full(num_spectrograms_per_subject, score_10k)
    seconds_10k_array = np.full(num_spectrograms_per_subject, seconds_10k)
    subject_id_array = np.full(num_spectrograms_per_subject, filename_and_id)
    filedir = dir_spectrogram + tread_or_overground + '/' + sensor
    
    if save_flag:
        count = prep.write_spectrograms_to_tfr(spectrogram_images, score_10k_array, seconds_10k_array, filedir, subject_id_array)
    
                

    