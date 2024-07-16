#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 18:59:59 2022

@author: patrickmayerhofer

create_TFRecords3

This file creates TFRecords. From CSV to Parquet to TFRecord

Based on:
https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c"""



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

data_sampling_fq = 500

# changeable variables
variables = ['left_ax', 'left_ay', 'left_az', 'left_gx', 'left_gy', 'left_gz', 
               'right_ax', 'right_ay', 'right_az', 'right_gx', 'right_gy', 'right_gz', 
               'subject_id', 'trial_id', 'speed', 'tread or overground', 'score_10k']
y_mock_variable = ['left_ax']
time_steps = 10000
step = 5000
nperseg=500 
noverlap=462
num_samples = 12 # each window is a sample. will save multiple windows in a tfrecord
#tread_or_overground = 'tread'

subjects = [179] #range(1,2)
tread_or_overground = 'Treadmill'
trials = [1]
speeds = [2.5,3.0,3.5]

# my helpful directories
dir_root = '/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Data/'
dir_prepared = dir_root + 'Prepared/'
dir_csv = dir_prepared + 'csv/'
dir_tfr = dir_prepared + "tfrecords/"
dir_spectrogram = dir_tfr + "windows_10000_spectrogram/"
dir_parquet = dir_prepared + "parquet/"
dir_data_overground = dir_csv + 'Overground/'
dir_data_treadmill = dir_csv + 'Treadmill/'
dir_data_tfr_overground = dir_spectrogram + 'Overground/'
dir_data_tfr_treadmill = dir_spectrogram + 'Treadmill/'

# get all directories 
all_files = np.sort(glob.glob(dir_csv + "*/*"))
#np.random.shuffle(files)

overview_file = pd.read_csv(dir_root + 'my_overview_file.csv')

for subject in subjects:
    ## create a dictionary and then dataframe with info for each file
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
            df_acc_and_angvel = v.iloc[:, 2:14]
            list_samples_cur = df_acc_and_angvel.to_dict('records')
            
            t_length = int((time_steps - nperseg) / (nperseg - noverlap) + 1)
            f_length = int((nperseg / 2) + 1)
            spectrogram_image = np.empty((1,t_length,f_length,12))
            for feature in range(len(df_acc_and_angvel.iloc[0])):
                f, t, Sxx = signal.spectrogram(df_acc_and_angvel.iloc[:,feature], data_sampling_fq, nperseg=nperseg, noverlap=noverlap)
                spectrogram_image[0,:,:,feature] = Sxx
                #plt.pcolormesh(t, f, Sxx, shading='gouraud')
                #plt.ylabel('Frequency [Hz]')
                #plt.xlabel('Time [sec]')
                #plt.show()
            
            
            dict_samples_cur = {
                'filename': filename_and_id + '_' + str(i),
                'fullpath': dir_file,
                'score_10k' : score_10k,
                #'channel_signals': list_samples_cur, # not needed for now, and uses too much energy and memory
                'spectrogram_image': spectrogram_image,
                'height' : spectrogram_image.shape[1],
                'width' : spectrogram_image.shape[2],
                'depth' : spectrogram_image.shape[3],
                'subject_id': subject_id,
                'tread_or_overground': tread_or_overground_bool
            } 
            file_id += 1
            
            output_list.append(dict_samples_cur)
                    
    # create dataframe    
    output_list_df = pd.DataFrame(output_list)
                
    """create TFRecords"""
    """Now save into tfrecords, where each tf record has 64 windows"""
    print("saving: subject " + str(subject_id))
    num_tfrecords = len(output_list_df) // num_samples
    print('10k_score: ' + str(output_list_df.score_10k.iloc[0]))
    if len(output_list) % num_samples:
        num_tfrecords += 1  # add one record if there are any remaining samples
    
    #print(num_tfrecords)
    
    #tfrecords_dir = '/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Code/time_series_tfrecords_example/working/tfrecords'
    
    if not os.path.exists(dir_spectrogram + tread_or_overground + '/' + sensor + '/'):
        os.makedirs(dir_spectrogram + tread_or_overground + '/' + sensor + '/')  # creating TFRecords output folder
    
    for tfrec_num in range(num_tfrecords):
        samples = output_list_df.loc[(tfrec_num * num_samples) : ((tfrec_num + 1) * num_samples)-1]
    
        with tf.io.TFRecordWriter(
            dir_spectrogram + tread_or_overground + '/' + sensor + "/" + sensor + "_%.2i-%i.tfrec" % (tfrec_num, len(samples))
        ) as writer:
            for index, row_sample in samples.iterrows():
                example = prep.create_example_spectrogram(row_sample)
                writer.write(example.SerializeToString())
                

    