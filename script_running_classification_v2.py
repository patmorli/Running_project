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
subjects = [1,2,3,4] # problems:  3 (seems to have two datasets twice)
#subjects = list(range(1,11))
speeds = [2.5, 3.0, 3.5] #[2.5, 3.0, 3.5]
trials = [1]
treadmill_flag = 1
overground_flag = 1
#plot_flag_treadmill = 0
#plot_flag_overground = 1
save_flag = 0

time_steps = 5000
step = 2500 

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
    # delete non-usable data
    data_treadmill_cut, data_treadmill_key = frc.import_cut_treadmill(subjects, trials, speeds, dir_data_treadmill)
    
    # get windows of length time_steps with a window slide of length step
    windows_treadmill_X, windows_treadmill_y = frc.get_treadmill_windows(X_variables, y_variable, data_treadmill_cut, time_steps, step)    
    
if overground_flag:
    # delete non-usable data
    data_overground_cut, data_overground_key = frc.import_cut_overground(subjects, trials, dir_data_overground)

    # get windows of length time_steps with a window slide of length step
    windows_overground_X, windows_overground_y = frc.get_overground_windows(X_variables, y_variable, data_overground_cut, time_steps, step)


  
"""split in training and test - later this should be multiple test and training sets"""
if y_variable == "subject_id":
    ## find indexes of changing id
    index_changing_ids = np.where(windows_treadmill_y[:-1] != windows_treadmill_y[1:])[0]
    train_test_index_treadmill = []
    
    ## create train and test set
    for i in range(0, len(index_changing_ids)+1):
        # create array from 0 - number of windows and shuffle array
        if i == 0: # for first iteration
            l = index_changing_ids[i]+1 # length of the data of this subject
        elif i == len(index_changing_ids): # for last iteration
            l = len(windows_treadmill_y) - (index_changing_ids[i-1] + 1) # length of the data of this subject
        else:
            l = index_changing_ids[i] - index_changing_ids[i-1] # length of the data of this subject
        index_treadmill = list(range(0, l))
        np.random.shuffle(index_treadmill)
        
        # create one index where 0s are test and 1s are training
        index_treadmill_70 = round(len(index_treadmill)*0.7)
        train_index_treadmill = index_treadmill[0:index_treadmill_70]
        train_test_index = np.zeros(l)
        train_test_index[train_index_treadmill] = 1
        
        # add index to overall index
        train_test_index_treadmill = np.append(train_test_index_treadmill, train_test_index)
     
    ## put it all together and shuffle data
    train_test_index_treadmill = train_test_index_treadmill.astype(bool)    
    train_X = windows_treadmill_X[train_test_index_treadmill]
    train_y = windows_treadmill_y[train_test_index_treadmill]  
    index_treadmill = list(range(0, len(train_y)))
    np.random.shuffle(index_treadmill)
    train_X = train_X[index_treadmill]
    train_y = train_y[index_treadmill]
    
    
    test_X = windows_treadmill_X[train_test_index_treadmill == False]
    test_y = windows_treadmill_y[train_test_index_treadmill == False]
    index_treadmill = list(range(0, len(test_y)))
    np.random.shuffle(index_treadmill)
    test_X = test_X[index_treadmill]
    test_y = test_y[index_treadmill]
    
    
    
    
            
if y_variable == "tread or overground":
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

pred_y = model.predict(test_X)
test_y_decoded = enc.inverse_transform(test_y)
pred_y_decoded = enc.inverse_transform(pred_y)

frc.plot_cm(test_y_decoded, pred_y_decoded, enc.categories_)


scores = model.evaluate(test_X, test_y, verbose=0)


