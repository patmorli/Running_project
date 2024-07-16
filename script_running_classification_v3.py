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
overground_flag = 0
#plot_flag_treadmill = 0
#plot_flag_overground = 1
save_flag = 0

time_steps = 5000
step = 2500 

X_variables = ['left_ax', 'left_ay', 'left_az', 'left_gx', 'left_gy', 'left_gz', 
               'right_ax', 'right_ay', 'right_az', 'right_gx', 'right_gy', 'right_gz']
y_variable = 'subject_id'
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


  
"""split in training and test"""
if y_variable == "subject_id":
    # save results for later in these variables
    pred_y_decoded = list()
    test_y_decoded = list()
    scores = list()
    
    for i in range(0, 6): # 6-fold testing
        # code asumes that each subject has 24 windows (if we only use trial 1)
        if len(windows_treadmill_y[0]) != 24:
            print("STOOOOOOOP!!!! Data should have length 24 in each subject. ")
            exit()
        
        ## create train_test_data. 1 is test data, 0 is train data  
        # first decide which windows in each subject we use 
        # (e.g.: in the first iteration the test windows are 0,1,2,3 of each subject)
        test_data = list(range(4*i,4*i+4))
        train_test_data = np.zeros(24)
        train_test_data[test_data] = 1
        train_test_data = train_test_data.astype(bool)
        
        # create empty variables
        train_X = np.empty([0, time_steps, len(X_variables)])
        train_y = np.empty([0, 1])
        test_X = np.empty([0, time_steps, len(X_variables)])
        test_y = np.empty([0, 1])
        
        # fill these variables for each subject
        # len(windows_treadmill_y) is same size as subjects
        for s in range(0, len(windows_treadmill_y)): 
            test_y = np.append(test_y, windows_treadmill_y[s][train_test_data])
            test_X = np.append(test_X, windows_treadmill_X[s][train_test_data], axis = 0)
            train_y = np.append(train_y, windows_treadmill_y[s][train_test_data == False])
            train_X = np.append(train_X, windows_treadmill_X[s][train_test_data == False], axis = 0)
        
        """now we need to change this part so that it can save all results from the for loop"""
        # hot encoder
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        enc = enc.fit(test_y.reshape(-1,1))
        
        
        train_y = enc.transform(train_y.reshape(-1,1))
        test_y = enc.transform(test_y.reshape(-1,1))
        
        """build and train model"""
        model = frc.create_model(train_X, train_y)
        history = model.fit(
                train_X, train_y,
                epochs=10,
                #validation_split=0.1
            )
        
        pred_y = model.predict(test_X)
        test_y_decoded.append(enc.inverse_transform(test_y))
        pred_y_decoded.append(enc.inverse_transform(pred_y))
        
        #frc.plot_cm(test_y_decoded[i], pred_y_decoded[i], enc.categories_)
        
        
        scores.append(model.evaluate(test_X, test_y, verbose=0))
        
        """add F1 score here and then plot it after everything is done!!!"""
        
  
    
    
    
    
            
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


    



