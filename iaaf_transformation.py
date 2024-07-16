#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:49:04 2022

@author: patrickmayerhofer

iaaf_transformation

This file takes the best times of the subjects and translates them into a mercier score
"""

import pandas as pd
import re
import math
import numpy as np

dir_root = '/Users/patrickmayerhofer/Google Drive/My Drive/Running Plantiga Project/'
filename = 'Subject_questionnaire.csv'
save_flag = 0

subject_file = pd.read_csv(dir_root + filename)
iaaf_score_men = pd.read_csv(dir_root + 'IAAF Scores/Men_Organized.csv')
iaaf_score_women = pd.read_csv(dir_root + 'IAAF Scores/Women_Organized.csv')


"""get all important information organized into one file"""
# find all empty rows
subject_file = subject_file[subject_file['Q1.3'].notnull()]
# find times of runners
genders = subject_file.iloc[2:, 20]
times_m5000 = subject_file.iloc[2:, 60]
times_m10000 = subject_file.iloc[2:, 61]
times_halfmarathon = subject_file.iloc[2:, 62]
times_marathon = subject_file.iloc[2:, 63]

# reset indexes
genders = genders.reset_index(drop= True)
times_m5000 = times_m5000.reset_index(drop= True)
times_m10000 = times_m10000.reset_index(drop= True)
times_halfmarathon = times_halfmarathon.reset_index(drop= True)
times_marathon = times_marathon.reset_index(drop= True)


"""Convert to seconds"""
## 10000m
times = times_m10000
seconds = np.empty(len(times))

# 21:00, 21:00:00, 21:00.00, 00:21:00, 00:21, O0:21, 21.00, 21 min, 21 m, 21 minutes, (O0:21 - not at 10k?), 
# 21.00, 00.21, 21', nan, N/a, 21m, 21min, 21minutes, :21, 1:02, :21, 3805 (??)

# 10000m
def k10_to_seconds(times):
    for i in range(0, len(times)):
        # all but nan, N/a
        try: 
            split_time = re.split('[:.]', times[i])
            
            # 21min, 21, 21minutes, 21m, 21', 21 m, 21 mins, 21 minutes,
            if len(split_time) == 1:
                
                if len(split_time[0]) > 2 and split_time[0].isdigit():
                    seconds[i] = float("nan")
                else:
                    seconds[i] = float(split_time[0][0:2])*60
            
            # 21:00, 21.00, 00.21, 00:21, :21
            elif len(split_time) == 2:
                 # :21
                if not split_time[0]:
                    seconds[i] = float(split_time[1])*60
                # 1:02    
                elif len(split_time[0]) == 1:
                    seconds[i] = float(split_time[0])*60*60 + float(split_time[1])*60
                    
                # 00:21, 00.21
                elif float(split_time[0]) == 0:
                    seconds[i] = float(split_time[1])*60
                    
                else:
                    #21:00, 21.00
                    try:
                        seconds[i] = float(split_time[0])*60 + float(split_time[1])
                    # 21 mins, 21 m, 21 minutes
                    except:
                        seconds[i] = float(split_time[0])*60
            
            # 21:00:00, 21:00.00, 00:21:00, 00:21.00
            elif len(split_time) == 3:
                # 00:21:00, 00:21.00
                if split_time[0] == 0:
                    seconds[i] = float(split_time[1])*60 + float(split_time[2])
                # 21:00:00, 21:00.00
                else:
                    seconds[i] = float(split_time[0])*60 + float(split_time[1])
        # nan and n/a
        except:
            seconds[i] = float("nan")
        
    return seconds
        
seconds_10k = k10_to_seconds(times)



def half_and_marathon_to_seconds(times):    
    ## halfmarathon and marathon
    seconds = np.empty(len(times))
    for i in range(0, len(times)):
        try:
            # get numbers 
            split_time1 = re.split('(\d+)', times[i])
            split_time = list()
            for s in range(0, len(split_time1)):
                if split_time1[s].isdigit():
                    split_time.append(split_time1[s])
            # 135 min, 2h ,..
            if len(split_time) == 1:
                if float(split_time[0]) == 0:
                    seconds[i] = float("nan")
                elif len(split_time[0]) > 3:
                    seconds[i] = float("nan")
                elif len(split_time[0]) > 1:
                   seconds[i] = float(split_time[0])*60
                else:
                   seconds[i] = float(split_time[0])*60*60
            # 1:18, 1hr 38mins,,...        
            elif len(split_time) == 2:
               seconds[i] = float(split_time[0])*60*60 + float(split_time[1])*60
             
            # 1:49:38, 1hr 49mins 38s, 87:20:00...   
            elif len(split_time) == 3:
                if len(split_time[0])>1:
                    seconds[i] = float(split_time[0])*60 + float(split_time[1])
                else:
                    seconds[i] = float(split_time[0])*60*60 + float(split_time[1])*60 + float(split_time[2])
            # n/a
            else:
                seconds[i] = float("nan")
        except:
            seconds[i] = float("nan")
    return seconds
       
    
times = times_halfmarathon
seconds_halfmarathon = half_and_marathon_to_seconds(times)
times = times_marathon
seconds_marathon = half_and_marathon_to_seconds(times)

if save_flag:
    d = {'seconds_10k': seconds_10k, 'seconds_halfmarathon': seconds_halfmarathon, 'seconds_marathon': seconds_marathon}
    df = pd.DataFrame(data=d)
    df.to_csv(dir_root + 'Subject_seconds.csv')
  


"""save score for each subject"""
scores_10k = np.empty(len(seconds_10k))
for i in range(0, len(seconds_10k)):
    if np.isnan(seconds_10k[i]):
        scores_10k[i] = seconds_10k[i]
    else:
        if genders[i] == "Female":
            index = iaaf_score_women['sec_10 km'].sub(seconds_10k[i]).abs().idxmin()
            scores_10k[i] = iaaf_score_women.loc[index, ['Points']]
            if scores_10k[i] == 1:
                print(['Subject ' + str(i)])
                print(genders[i])
                print(str(seconds_10k[i]))
                
        elif genders[i] == "Male":
            index = iaaf_score_men['sec_10 km'].sub(seconds_10k[i]).abs().idxmin()  
            scores_10k[i] = iaaf_score_men.loc[index, ['Points']]
            if scores_10k[i] == 1:
                print(['Subject ' + str(i)])
                print(genders[i])
                print(str(seconds_10k[i]))
                
                
# Just copied and pasted the scores_10k into subject_seconds.csv
   
   
        
    
        
    
    
    

