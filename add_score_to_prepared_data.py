#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:49:55 2022

@author: patrickmayerhofer

add_score_to_prepared_data

required files:
- prepared .csv files of participants
- overview file 


How it works:
- This script adds the 10k IAAF score to each participant's running file'
"""

import pandas as pd
import numpy as np

subjects = [1,2,3,8,9,15,20,21,24,28,32,33,35,36,41,42,45,49,55,59,79,108,110,111,122,131,133] #[5,6,7,8,9,10] # problems:  3 (seems to have two datasets twice)
save_flag = 1

# my helpful directories
dir_root = '/Users/patrick/Google Drive/My Drive/Running Plantiga Project/'
dir_data_prepared = dir_root + 'Data/Prepared/csv/'
dir_overview_file = dir_root + 'Data/my_overview_file.csv'
dir_data_treadmill = dir_data_prepared + 'Treadmill/'
dir_data_overground = dir_data_prepared + 'Overground/'
dir_scores_file = dir_root + 'Data/Subject_seconds_scores.csv'

"""import overview file and score file"""
overview_file = pd.read_csv(dir_overview_file)
score_file = pd.read_csv(dir_scores_file)

data_treadmill_key = list('0') # index place 0 with 0 to store stubject 1 at index 1
data_treadmill_cut = list('0')
   
"""add scores to each prepared csv file"""

#for i in range(0, len(overview_file)):
for subject in subjects:
    print('running subject ' + str(subject))
    i = subject - 1
    # for each column 
    overview_columns = overview_file.columns
    # delete subject_id column
    overview_columns = overview_columns[1:13]
    
    for column in overview_columns:
        key = overview_file.loc[i, [column]]
        key = key[column]
        # if it is not string, it is float because nan
        if type(key) == str:
            try:
                #import
                if column == 'overground' or column == 'overground_2' or column == 'overground_3':
                    data = pd.read_csv(dir_data_overground + str(key) + '.csv')
                else:
                    data = pd.read_csv(dir_data_treadmill + str(key) + '.csv') 
                 
                # try, in case there is already a scores_10k field    
                try:  
                    ## add score to data
                    score = np.zeros(len(data)) + float(score_file.loc[i, ['scores_10k']])
                    data.insert(17,"score_10k", score)
                    
                    ## save file
                    if column == 'overground' or column == 'overground_2' or column == 'overground_3':
                        data.to_csv(dir_data_overground + str(key) + '.csv')
                    
                    else:
                        data.to_csv(dir_data_treadmill + str(key) + '.csv')
                except:
                    print(key + ' already has a scores_10k field bro')
            except:
                print(key + ' did not work.')
        else:
            print('key is nan')