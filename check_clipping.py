#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 17:21:28 2022

@author: patrickmayerhofer

check_clipping
"""

import pandas as pd
import matplotlib.pyplot as plt


# changeable variables
subjects = [4] # problems:  3 (seems to have two datasets twice)
#subjects = list(range(1,11))
speeds = [2.5] #[2.5, 3.0, 3.5]
trials = [1]
treadmill_flag = 1
overground_flag = 0

#data_id = '184_3.0'

tread_or_over = 'Treadmill'

# my helpful directories
dir_root = '/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Data/Prepared/'

if treadmill_flag:
    for subject_id in subjects:
        for speed in speeds:
            for trial_id in trials:
                if trial_id == 1:
                    data_key = '/SENSOR' + str(subject_id).zfill(3) + '_' + str(speed) + '.csv'
                else:
                    data_key = '/SENSOR' + str(subject_id).zfill(3) + '_' + str(speed) + '_' + str(trial_id) + '.csv'
                    
                dir_data = dir_root + 'Treadmill'  + data_key
              
                #import
                data = pd.read_csv(dir_data)
                
                #plot
                plt.figure()
                plt.title(data_key)
                raw_left_ax = plt.plot(data.left_ax, label = 'raw_left_ax', color = 'b')
                raw_left_ay = plt.plot(data.left_ay, label = 'raw_left_ay', color = 'b')
                raw_left_az = plt.plot(data.left_az, label = 'raw_left_az', color = 'b')
                new_left_ax = plt.plot(data.left_ax[(data['usabledata'] == True)], label = 'raw_left_ax', color = 'r')
                new_left_ay = plt.plot(data.left_ay[(data['usabledata'] == True)], label = 'raw_left_ay', color = 'r')
                new_left_az = plt.plot(data.left_az[(data['usabledata'] == True)], label = 'raw_left_az', color = 'r')
                plt.legend()
                
if overground_flag:
    for subject_id in subjects:
            for trial_id in trials:
                if trial_id == 1:
                    data_key = '/SENSOR' + str(subject_id).zfill(3) + '_run' + '.csv'
                else:
                    data_key = '/SENSOR' + str(subject_id).zfill(3) + '_run' + '_'  + str(trial_id) + '.csv'
                    
                dir_data = dir_root + 'Overground'  + data_key
              
                #import
                data = pd.read_csv(dir_data)
                
                #plot
                plt.figure()
                plt.title(data_key)
                raw_left_ax = plt.plot(data.left_ax, label = 'raw_left_ax', color = 'b')
                raw_left_ay = plt.plot(data.left_ay, label = 'raw_left_ay', color = 'b')
                raw_left_az = plt.plot(data.left_az, label = 'raw_left_az', color = 'b')
                new_left_ax = plt.plot(data.left_ax[(data['usabledata'] == True)], label = 'raw_left_ax', color = 'r')
                new_left_ay = plt.plot(data.left_ay[(data['usabledata'] == True)], label = 'raw_left_ay', color = 'r')
                new_left_az = plt.plot(data.left_az[(data['usabledata'] == True)], label = 'raw_left_az', color = 'r')
                plt.legend()