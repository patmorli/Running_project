#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 15:18:01 2022

@author: patrickmayerhofer

script_running_data_preparation

needs: functions_running_data_preparation

- Loads raw running data for treadmill and/or overground data
- Cuts the treadmill data to 25k long datasets per speed (3) and trial (1-3), deleting acceleration and deceleration (labels usable and non-usable data)
- Cuts the acceleration and deceleration in the data to have constant data (labels usable and non-usable data)
- Plots the results, to double-check if algorithm worked properly. If not, manually change what is usable and non-usable data
- Saves data to csv files
"""


# my helpful libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bottleneck as bn
import sys 
sys.path.append('/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Code/')
import functions_running_data_preparation as frg
import pickle
import os




# changeable variables
subjects = [1,2,3,8,9,20,21,24,28,32,33,35,36,42,45,49,55,79,108,110,111,122,131,133] # problems:  3 (seems to have two datasets twice)
#subjects = list(range(1,11))
speeds = [2.5, 3.0, 3.5] #[2.5, 3.0, 3.5]
trials = [3]
treadmill_flag = 0
overground_flag = 1
plot_flag_treadmill = 1
plot_flag_overground = 1
save_flag = 0
save_overview_flag = 0
x_debugging = 0 # for the subjects that have _x at the end (data was stolen, that had nothing to do with our data and analysis)

#trials = ['2.5', '2.5_2', '2.5_3', '3.0', '3.0_2', '3.0_3', '3.5', '3.5_2', '3.5_3']

# my helpful directories
dir_root = '/Users/patrick/Google Drive/My Drive/Running Plantiga Project/'
dir_data_raw = dir_root + 'Data/Raw/'
dir_overview_file = dir_root + 'Data/my_overview_file.csv'

"""import overview file"""
overview_file = pd.read_csv(dir_overview_file)


"""import raw data and add subject id, trial id, and speeds to each dataset"""
"""mark unusable data from treadmill running and/or overground running with 0 and rest with 1""" 
"""save data"""
if treadmill_flag:
    data_treadmill_raw_key, data_treadmill_raw, overview_file = frg.import_treadmill_data(dir_data_raw, subjects, speeds, trials, overview_file, x_debugging)
    data_treadmill_raw = frg.clean_treadmill_data(data_treadmill_raw, data_treadmill_raw_key, plot_flag_treadmill)
    
    
    """save each subjects's data with the key as its name"""
    if save_flag:
        for i in range(1,len(data_treadmill_raw_key)):
            for z in range(0,len(data_treadmill_raw_key[i])):
                for a in range(0,len(data_treadmill_raw_key[i][z])):
                    if data_treadmill_raw_key[i][z][a] != 'not':
                        filepath = dir_root + 'Data/Prepared/csv/Treadmill/' 
                        if os.path.isdir(filepath) != True:
                            os.makedirs(filepath)
                        file = data_treadmill_raw[i][z][a]
                        file.to_csv(filepath + data_treadmill_raw_key[i][z][a] + '.csv', index = False)
        
    
    

 
    
    
if overground_flag:
    data_overground_raw_key, data_overground_raw, overview_file = frg.import_overground_data(dir_data_raw, subjects, trials, overview_file, x_debugging)
    data_overground_raw = frg.clean_overground_data(data_overground_raw, data_overground_raw_key, plot_flag_treadmill)
    
    
    """save each subjects's data with the key as its name"""
    if save_flag:
        for i in range(1,len(data_overground_raw_key)):
            for z in range(0,len(data_overground_raw_key[i])):
                    if data_overground_raw_key[i][z] != 'not':
                        filepath = dir_root + 'Data/Prepared/csv/Overground/'
                        if os.path.isdir(filepath) != True:
                            os.makedirs(filepath)
                        file = data_overground_raw[i][z]
                        file.to_csv(filepath + data_overground_raw_key[i][z] + '.csv', index = False)

# also save overview file
if save_overview_flag:
    overview_file.to_csv(dir_overview_file, index = False)    
       


