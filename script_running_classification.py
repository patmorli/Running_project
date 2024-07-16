#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:22:47 2022

@author: patrickmayerhofer

script_running_classification
"""

# my helpful libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bottleneck as bn
import sys 
sys.path.append('/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Code/')
import functions_running_classification as frc
import pickle
from sklearn.preprocessing import OneHotEncoder 


# changeable variables
subjects = [1] # problems:  3 (seems to have two datasets twice)
#subjects = list(range(1,11))
speeds = [2.5, 3.0, 3.5] #[2.5, 3.0, 3.5]
trials = [1, 2, 3]
treadmill_flag = 1
overground_flag = 1
#plot_flag_treadmill = 0
#plot_flag_overground = 1
save_flag = 0

X_variables = ['left_ax', 'left_ay', 'left_az', 'left_gx', 'left_gy', 'left_gz', 
               'right_ax', 'right_ay', 'right_az', 'right_gx', 'right_gy', 'right_gz']
y_variable = 'tread or overground'
layers_nodes = [8,8,8]
learning_rate = 0.01

# my helpful directories
dir_root = '/Volumes/GoogleDrive/My Drive/Running Plantiga Project/'
dir_data_prepared = dir_root + 'Data/Prepared/'
dir_overview_file = dir_root + 'Data/my_overview_file.csv'
dir_data_treadmill = dir_data_prepared + 'Treadmill/'
dir_data_overground = dir_data_prepared + 'Overground/'

"""import overview file"""
overview_file = pd.read_csv(dir_overview_file)

"""import data, delete unusable data, and prepare classification windows"""
   
if treadmill_flag:
    """Treadmill import and cut out usable data"""
    # creates lists of lists of lists (3 layers)
    # data_treadmill_key holds the names of each dataset from data_treadmill_cut
    # first layer: subjects, second layer: trials, third layer: speeds, fourth layer: actual data

    data_treadmill_key = list('0') # index place 0 with 0 to store stubject 1 at index 1
    data_treadmill_cut = list('0')
    
    for subject_id in subjects:
        subject_name = 'SENSOR' + str(subject_id).zfill(3)
        data_cut_trials = list()
        data_key_trials = list()
        for trial_id in trials:
            data_cut_speeds = list()
            data_key_speeds = list()
            for speed in speeds:
                if trial_id == 1:
                    key = subject_name + '_' + str(speed)
                else:
                    key = subject_name + '_' + str(speed) + '_' + str(trial_id)
        
                # import dataset
                data = pd.read_csv(dir_data_treadmill + key + '.csv')
                # delete unusable data (usabledata = False)
                data = data[data['usabledata'] == True]
                
                data_cut_speeds.append(data)
                data_key_speeds.append(key) 
                
            data_key_trials.append(data_key_speeds)
            data_cut_trials.append(data_cut_speeds)
    
        data_treadmill_key.append(data_key_trials)
        data_treadmill_cut.append(data_cut_trials)
        
    ## data to use
    windows_treadmill_X = np.empty([0, time_steps, len(X_variables)])
    windows_treadmill_y = np.empty([0, 1])
    for subject_id in range(1,len(data_treadmill_cut)):
        for trial_id in range(0, len(data_treadmill_cut[subject_id])):
            for speed in range(0, len(data_treadmill_cut[subject_id][trial_id])):
                data_treadmill_X = data_treadmill_cut[subject_id][trial_id][speed].loc[:, X_variables]
                data_treadmill_y = data_treadmill_cut[subject_id][trial_id][speed].loc[:, y_variable]
                
                X, y =  \
                        frc.create_dataset(data_treadmill_X, data_treadmill_y, \
                                           time_steps=time_steps, step=step)  
                            
                windows_treadmill_X = np.append(windows_treadmill_X, X, axis = 0)
                windows_treadmill_y = np.append(windows_treadmill_y, y, axis = 0)

  
      
if overground_flag:
    """overground import and cut to usable data"""
    # creates lists of lists of lists (3 layers)
    # data_overground_raw_key holds the names of each dataset from data_overground_raw
    # first layer: subjects, second layer: trials, third layer: both directions
    data_overground_key = list('0') # index place 0 with 0 to store stubject 1 at index 1
    data_overground_cut = list('0')
    for subject_id in subjects:
        subject_name = 'SENSOR' + str(subject_id).zfill(3)
        data_cut_trials = list()
        data_key_trials = list()
        
        for trial_id in trials:
            if trial_id == 1:
                key = subject_name + '_run'
            else:
                key = subject_name + '_run' + '_' + str(trial_id)
        
            
            # import dataset
            data = pd.read_csv(dir_data_overground + key + '.csv')
            # create two datasets, for two directions ran
            p = 0
            for i in range(0, len(data)):
                if data.loc[i,'usabledata'] == True and p == 0:
                    data_1_start = i
                    p = 1
                
                if data.loc[i,'usabledata'] == False and p == 1:
                    data_1_end = i
                    p = 2
                
                if data.loc[i,'usabledata'] == True and p == 2:
                    data_2_start = i
                    p = 3
                    
                if data.loc[i,'usabledata'] == False and p == 3:
                    data_2_end = i
                    break
             
            # from now on, overground data has 3 dimensions too. the third are the two directions.     
            data_split = list()    
            data_split.append(data[data_1_start:data_1_end])
            data_split.append(data[data_2_start:data_2_end])
            
                
            data_key_trials.append(key)      
            data_cut_trials.append(data_split)
    
        data_overground_key.append(data_key_trials)
        data_overground_cut.append(data_cut_trials)

if y_variable == 'tread or overground':
    """prepare train and test data"""
    """split data into windows of same length"""
    time_steps = 5000
    step = 2500 
    
    """treadmill"""
    ## data to use
    
    windows_treadmill_X = np.empty([0, time_steps, len(X_variables)])
    windows_treadmill_y = np.empty([0, 1])
    for subject_id in range(1,len(data_treadmill_cut)):
        for trial_id in range(0, len(data_treadmill_cut[subject_id])):
            for speed in range(0, len(data_treadmill_cut[subject_id][trial_id])):
                data_treadmill_X = data_treadmill_cut[subject_id][trial_id][speed].loc[:, X_variables]
                data_treadmill_y = data_treadmill_cut[subject_id][trial_id][speed].loc[:, y_variable]
                
                X, y =  \
                        frc.create_dataset(data_treadmill_X, data_treadmill_y, \
                                           time_steps=time_steps, step=step)  
                            
                windows_treadmill_X = np.append(windows_treadmill_X, X, axis = 0)
                windows_treadmill_y = np.append(windows_treadmill_y, y, axis = 0)

    """overground"""
    windows_overground_X = np.empty([0, time_steps, len(X_variables)])
    windows_overground_y = np.empty([0, 1])
    for subject_id in range(1,len(data_overground_cut)):
        for trial_id in range(0, len(data_overground_cut[subject_id])):
            for direction in range(0, len(data_overground_cut[subject_id][trial_id])):
                data_overground_X = data_overground_cut[subject_id][trial_id][direction].loc[:, X_variables]
                data_overground_y = data_overground_cut[subject_id][trial_id][direction].loc[:, y_variable]
               
                X, y =  \
                        frc.create_dataset(data_overground_X, data_overground_y, \
                                           time_steps=time_steps, step=step)  
                            
                windows_overground_X = np.append(windows_overground_X, X, axis = 0)
                windows_overground_y = np.append(windows_overground_y, y, axis = 0)



  
    """split in training and test - later this should be multiple test and training sets"""
    ## shuffle data 
    # create array from 0 - number of windows and shuffle array
    index_treadmill = list(range(0,len(windows_treadmill_y)))
    np.random.shuffle(index_treadmill)
    index_overground = list(range(0,len(windows_overground_y)))
    np.random.shuffle(index_overground)
    
    # create index for train and test
    index_treadmill_70 = round(len(index_treadmill)*0.7)
    train_index_treadmill = index_treadmill[0:index_treadmill_70]
    test_index_treadmill = index_treadmill[index_treadmill_70:len(index_treadmill)]
    
    index_overground_70 = round(len(index_overground)*0.7)
    train_index_overground = index_overground[0:index_overground_70]
    test_index_overground = index_overground[index_overground_70:len(index_overground)]
    
    ## create train and test set
    train_X = np.concatenate((windows_treadmill_X[train_index_treadmill], windows_overground_X[train_index_overground]), axis = 0)
    train_y = np.concatenate((windows_treadmill_y[train_index_treadmill], windows_overground_y[train_index_overground]), axis = 0)

    test_X = np.concatenate((windows_treadmill_X[test_index_treadmill], windows_overground_X[test_index_overground]), axis = 0)
    test_y = np.concatenate((windows_treadmill_y[test_index_treadmill], windows_overground_y[test_index_overground]), axis = 0)
    
    # hot encoder
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(train_y)

    
    train_y = enc.transform(train_y)
    test_y = enc.transform(test_y)

    """build and train model"""
    model = frc.create_model(train_X, train_y)
    history = model.fit(
            train_X, train_y,
            epochs=10,
            #validation_split=0.1
        )

if y_variable == 'subject_id':
    """prepare train and test data"""
    
    """split data into windows of same length"""
    time_steps = 5000
    step = 2500 
    
    if treadmill_flag: 
        ## data to use
        windows_treadmill_X = np.empty([0, time_steps, len(X_variables)])
        windows_treadmill_y = np.empty([0, 1])
        for subject_id in range(1,len(data_treadmill_cut)):
            for trial_id in range(0, len(data_treadmill_cut[subject_id])):
                for speed in range(0, len(data_treadmill_cut[subject_id][trial_id])):
                    data_treadmill_X = data_treadmill_cut[subject_id][trial_id][speed].loc[:, X_variables]
                    data_treadmill_y = data_treadmill_cut[subject_id][trial_id][speed].loc[:, y_variable]
                    
                    X, y =  frc.create_dataset(data_treadmill_X, data_treadmill_y, \
                                               time_steps=time_steps, step=step)  
                                
                    windows_treadmill_X = np.append(windows_treadmill_X, X, axis = 0)
                    windows_treadmill_y = np.append(windows_treadmill_y, y, axis = 0)

    """overground"""
    windows_overground_X = np.empty([0, time_steps, len(X_variables)])
    windows_overground_y = np.empty([0, 1])
    for subject_id in range(1,len(data_overground_cut)):
        for trial_id in range(0, len(data_overground_cut[subject_id])):
            for direction in range(0, len(data_overground_cut[subject_id][trial_id])):
                data_overground_X = data_overground_cut[subject_id][trial_id][direction].loc[:, X_variables]
                data_overground_y = data_overground_cut[subject_id][trial_id][direction].loc[:, y_variable]
               
                X, y =  \
                        frc.create_dataset(data_overground_X, data_overground_y, \
                                           time_steps=time_steps, step=step)  
                            
                windows_overground_X = np.append(windows_overground_X, X, axis = 0)
                windows_overground_y = np.append(windows_overground_y, y, axis = 0)



  
    """split in training and test - later this should be multiple test and training sets"""
    ## shuffle data 
    # create array from 0 - number of windows and shuffle array
    index_treadmill = list(range(0,len(windows_treadmill_y)))
    np.random.shuffle(index_treadmill)
    index_overground = list(range(0,len(windows_overground_y)))
    np.random.shuffle(index_overground)
    
    # create index for train and test
    index_treadmill_70 = round(len(index_treadmill)*0.7)
    train_index_treadmill = index_treadmill[0:index_treadmill_70]
    test_index_treadmill = index_treadmill[index_treadmill_70:len(index_treadmill)]
    
    index_overground_70 = round(len(index_overground)*0.7)
    train_index_overground = index_overground[0:index_overground_70]
    test_index_overground = index_overground[index_overground_70:len(index_overground)]
    
    ## create train and test set
    train_X = np.concatenate((windows_treadmill_X[train_index_treadmill], windows_overground_X[train_index_overground]), axis = 0)
    train_y = np.concatenate((windows_treadmill_y[train_index_treadmill], windows_overground_y[train_index_overground]), axis = 0)

    test_X = np.concatenate((windows_treadmill_X[test_index_treadmill], windows_overground_X[test_index_overground]), axis = 0)
    test_y = np.concatenate((windows_treadmill_y[test_index_treadmill], windows_overground_y[test_index_overground]), axis = 0)
    
    # hot encoder
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(train_y)

    
    train_y = enc.transform(train_y)
    test_y = enc.transform(test_y)

    """build and train model"""
    model = frc.create_model(train_X, train_y)
    history = model.fit(
            train_X, train_y,
            epochs=10,
            #validation_split=0.1
        )
 