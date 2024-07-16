#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 18:59:59 2022

@author: patrickmayerhofer

create_TFRecords3

This file creates TFRecords. From CSV to Parquet to TFRecord

Based on:
https://www.kaggle.com/code/danmaccagnola/activity-recognition-data-w-tfrecords/notebook
"""



# my great libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import shutil
import sys 
sys.path.append('/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Code/')
import functions_running_data_preparation as prep
import tensorflow as tf
import glob
np.random.seed(1111)
from pathlib import Path 
import tensorflow as tf


# changeable variables
time_steps = 10000
step = 5000
num_samples = 12 # each window is a sample. will save multiple windows in a tfrecord
#tread_or_overground = 'tread'

subjects = [1,2,3,4,5,6,7,8,9,10] 
tread_or_overground = 'Treadmill'
trials = [1]
speeds = [2.5,3.0,3.5]
save_name = '10k_no_spectrogram'

# my helpful directories
dir_root = '/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Running Plantiga Project/Data/'
dir_prepared = dir_root + 'Prepared/'
dir_csv = dir_prepared + 'csv/'
dir_tfr = dir_prepared + "tfrecords/"
dir_parquet = dir_prepared + "parquet/"
dir_data_overground = dir_csv + 'Overground/'
dir_data_treadmill = dir_csv + 'Treadmill/'
dir_data_tfr_overground = dir_tfr + 'Overground/'
dir_data_tfr_treadmill = dir_tfr + 'Treadmill/'
dir_save = dir_tfr + save_name + '/' + tread_or_overground + '/' 


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
        seconds_10k = int(df_cur.seconds_10k.iloc[0])
        subject_id = int(df_cur.subject_id.iloc[0])
        tread_or_overground_bool = int(df_cur.iloc[0,18])
        if df_cur.trial_id.iloc[0] == 1:
            filename_and_id = 'SENSOR' + str(subject) + '_' + str(df_cur.speed.iloc[0]) + '.csv'  
        if df_cur.trial_id.iloc[0] == 2:
            filename_and_id = 'SENSOR' + str(subject) + '_' + str(df_cur.speed.iloc[0]) + '_2.csv'
        if df_cur.trial_id.iloc[0] ==3:
            filename_and_id = 'SENSOR' + str(subject) + '_' + str(df_cur.speed.iloc[0]) + '_3.csv'
            
                
                
        # creates windows and saves in a dataframe
        file_id = 0
        for i in range(0, len(df_cur)-time_steps + 1, step):
            v = df_cur.iloc[i:(i + time_steps)]
            df_acc_and_angvel = v.iloc[:, 3:15]
            list_samples_cur = df_acc_and_angvel.to_dict('records')
            dict_samples_cur = {
                'filename': filename_and_id,
                'fullpath': dir_file,
                'seconds_10k' : seconds_10k,
                'samples': list_samples_cur,
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
    print('10k_seconds: ' + str(output_list_df.seconds_10k.iloc[0]))
    """
    if len(output_list) % num_samples:
        num_tfrecords += 1  # add one record if there are any remaining samples
    """
    #print(num_tfrecords)
    
    #tfrecords_dir = '/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Code/time_series_tfrecords_example/working/tfrecords'
    
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)  # creating TFRecords output folder
    
    for tfrec_num in range(num_tfrecords):
        samples = output_list_df.loc[(tfrec_num * num_samples) : ((tfrec_num + 1) * num_samples)-1]
    
        with tf.io.TFRecordWriter(
            dir_save + sensor + ".tfrecords"
        ) as writer:
            for index, row_sample in samples.iterrows():
                example = prep.create_example(row_sample)
                writer.write(example.SerializeToString())
                

    