#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 19:02:07 2022

@author: patrickmayerhofer

functions_running_general
"""

import numpy as np
import bottleneck as bn
import scipy.signal as sps
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import pdb
import tensorflow as tf
import librosa
import librosa.display
from sklearn import preprocessing



def get_raw_data(df_cur, bin_subject_ids, subject, dir_file_number, time_steps, step, output_list, speed, plot_flag, flag_normalize_data):
    "Takes df_cur (12 features) as input and prepares data to save as tfrecord -- > put this in function" 
    # create output_list with moving windows
    score_10k = int(df_cur.score_10k.iloc[0])
    seconds_10k = int(df_cur.seconds_10k.iloc[0])
    subject_id = int(df_cur.subject_id.iloc[0])
    tread_or_overground_bool = int(df_cur.iloc[0,18])
    # find out which group the subject belongs to
    for bins in range(len(bin_subject_ids)):
        if subject in bin_subject_ids[bins]:
            my_bin = bins
            #print(my_bin)
            break
    
    """    
    # create filenames    
    if dir_file_number == 0:
        if df_cur.trial_id.iloc[0] == 1:
            filename_and_id = 'SENSOR' + str(subject) + '_' + str(df_cur.speed.iloc[0])
        if df_cur.trial_id.iloc[0] == 2:
            filename_and_id = 'SENSOR' + str(subject) + '_' + str(df_cur.speed.iloc[0]) + '_2'
        if df_cur.trial_id.iloc[0] ==3:
            filename_and_id = 'SENSOR' + str(subject) + '_' + str(df_cur.speed.iloc[0]) + '_3'
   
    # careful here, this is currently only optimized for trial 1
    else:
        filename_and_id = 'SENSOR' + str(subject)
    """    
    
    # creates windows and saves in a dataframe
    file_id = 0
    for i in range(0, len(df_cur)-time_steps + 1, step):
        v = df_cur.iloc[i:(i + time_steps)]
        df_acc_and_angvel = v.iloc[:, 3:15]
        
        if flag_normalize_data:
            #plt.figure()
            #plt.plot(df_acc_and_angvel.left_ax)
            
            scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            names = df_acc_and_angvel.columns
            d = scaler.fit_transform(df_acc_and_angvel)
            scaled_df = pd.DataFrame(d, columns=names)
            df_acc_and_angvel = scaled_df
            #plt.figure()
            #plt.plot(scaled_df.left_ax)
            
        if plot_flag:
            plt.figure()
            plt.plot(df_acc_and_angvel.left_ay)
            plt.title('Subject: '  + str(subject) + ', Speed: ' + str(speed))
        
        list_samples_cur = df_acc_and_angvel.to_dict('records')
        dict_samples_cur = {
            'samples': list_samples_cur,
            'bin_label': my_bin,
            'seconds_10k' : seconds_10k,
            'subject_id': subject_id,
            'tread_or_overground': tread_or_overground_bool,
            'speed': speed,
        } 
        file_id += 1
        output_list.append(dict_samples_cur)
    return output_list



def get_spectrograms(df_cur, subject, bin_subject_ids, dir_file_number, i_spectrogram, spectrogram_images, time_steps, step, signal, data_sampling_fq, nperseg, noverlap, plot_flag):
    "Takes df_cur (12 features) as input and creates a spectrogram for each feature"
    # create output_list with moving windows
    score_10k = int(df_cur.score_10k.iloc[0])
    seconds_10k = int(df_cur.seconds_10k.iloc[0])
    subject_id = int(df_cur.subject_id.iloc[0])
    tread_or_overground_bool = int(df_cur.iloc[0,18])
    
    # find out which group the subject belongs to
    for bins in range(len(bin_subject_ids)):
        if subject in bin_subject_ids[bins]:
            my_bin = bins
            break
        
    
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
                
                plt.figure()
                plt.plot(df_acc_and_angvel.iloc[:,0])
                
        i_spectrogram = i_spectrogram + 1  
    
    
    return spectrogram_images, score_10k, seconds_10k, subject_id, my_bin, i_spectrogram

def get_bins_info(subjects, overview_file, n_bins, bins_by_time_or_numbers, plot_bins_flag):
    """calculate bin ranges --> put this in a function
    also: save the subject_ids in each bin, we can use that later I think"""
   
    all_seconds10k_and_subject_id = overview_file[['subject_id', 'seconds_10k']].iloc[np.array(subjects)-1].reset_index(drop = 'true') #only includes eligible subjects
    
    
    
    # here, it creates bins of same time range
    if bins_by_time_or_numbers:
        all_seconds_10k = all_seconds10k_and_subject_id.seconds_10k
        max_seconds_10k = max(all_seconds_10k)
        min_seconds_10k = min(all_seconds_10k)
        all_subject_ids = all_seconds10k_and_subject_id.subject_id
        bin_size = round((max_seconds_10k-min_seconds_10k)/n_bins)
    
        # calculate ranges of bins
        bin_ranges = list()
        bin_ranges.append(min_seconds_10k)
        for i in range(n_bins):
            bin_ranges.append(min_seconds_10k + (i+1) * bin_size)
        
        # calculate how many values are in each bin
        bin_times = list()
        bin_subject_ids = list()
        for i in range(n_bins):
            all_times = list()
            all_groups = list()
            if i == n_bins - 1:
                for f in range(len(all_seconds_10k)):
                    if all_seconds_10k[f] >= bin_ranges[i]:
                        all_times.append(all_seconds_10k[f]) 
                        all_groups.append(all_subject_ids[f])
            else:
                for f in range(len(all_seconds_10k)):
                    if all_seconds_10k[f] >= bin_ranges[i] and all_seconds_10k[f] < bin_ranges[i+1]:
                        all_times.append(all_seconds_10k[f]) 
                        all_groups.append(all_subject_ids[f]) 
            bin_times.append(all_times)
            bin_subject_ids.append(all_groups)
                    
        if plot_bins_flag:
            plt.figure(100)
            for i in range(len(bin_times)):
                plt.plot(bin_times[i], marker = 'o', linestyle = '')
               
    
    # here, it creates bins that have the same amount of participants       
    else:
        bin_size = round(len(all_seconds10k_and_subject_id)/n_bins)
        all_seconds10k_and_subject_id = all_seconds10k_and_subject_id.sort_values(by = ['seconds_10k']).reset_index(drop = 'True')
        bin_times = list()
        bin_subject_ids = list()
        
        # put in their respective bins
        for i in range(n_bins):
            if i == n_bins - 1:
                bin_times.append(all_seconds10k_and_subject_id[i*bin_size:len(all_seconds10k_and_subject_id)].seconds_10k.reset_index(drop = 'True'))
                bin_subject_ids.append(all_seconds10k_and_subject_id[i*bin_size:len(all_seconds10k_and_subject_id)].subject_id.reset_index(drop = 'True'))
                
            else:
                bin_times.append(all_seconds10k_and_subject_id[i*bin_size:(i+1)*bin_size].seconds_10k.reset_index(drop = 'True'))
                bin_subject_ids.append(all_seconds10k_and_subject_id[i*bin_size:(i+1)*bin_size].subject_id.reset_index(drop = 'True'))
         
        # if a given time stamp is at the end of one bin and at the beginning of the next one as well, 
        # this will take the subject from one bin and put it in the other, so that same running seconds are always in the same bin
        for i in range(n_bins-1):  
            while bin_times[i+1][0] == bin_times[i][len(bin_times[i])-1]:
                bin_times[i][len(bin_times[i])] = bin_times[i+1][0]
                bin_times[i+1] = bin_times[i+1].drop([0]).reset_index(drop = True)
                
                bin_subject_ids[i][len(bin_subject_ids[i])] = bin_subject_ids[i+1][0]
                bin_subject_ids[i+1] = bin_subject_ids[i+1].drop([0]).reset_index(drop = True)
        
            
        if plot_bins_flag:
            plt.figure(100)
            for i in range(len(bin_times)):
                plt.plot(bin_times[i].sample(frac = 1).reset_index(drop = True), marker = 'o', linestyle = '')
     
        # convert series in list to list in list
        for i in range(len(bin_subject_ids)):
            bin_subject_ids[i] = bin_subject_ids[i].to_list()
        
    return bin_subject_ids , bin_times          

def import_treadmill_data(dir_data_raw, subjects, speeds, trials, overview_file, x_debugging):
    """import raw data and add subject id, trial id, and speeds to each dataset"""
    # creates lists of lists of lists (3 layers)
    # data_treadmill_raw_key holds the names of each dataset from data_treadmill_raw
    # first layer: subjects, second layer: trials, third layer: speeds, fourth layer: actual data
   
    
    data_treadmill_raw_key = list('0') # index place 0 with 0 to store stubject 1 at index 1
    data_treadmill_raw = list('0')
    for subject_id in subjects:
        subject_name = 'SENSOR' + str(subject_id).zfill(3)
        dir_data_raw_subject = dir_data_raw + subject_name + '/'
        data_raw_trials = list()
        data_raw_key_trials = list()
        for trial_id in trials:
            data_raw_speeds = list()
            data_raw_key_speeds = list()
            for speed in speeds:
                if trial_id == 1:
                    key = subject_name + '_' + str(speed)
                    overview_column_name = str(speed)
                else:
                    key = subject_name + '_' + str(speed) + '_' + str(trial_id)
                    overview_column_name = str(speed) + '_' + str(trial_id)
        
        
                # because not all subjects do have all runs, we try it, if there is an error, we add an empty list to the data and modify the key with a "not" 
                try:
                    if x_debugging:
                        key = key + '_x'
                        
                    data = pd.read_csv(dir_data_raw_subject + key + '.csv')
                    
                    # fill overview_file
                    overview_file.loc[subject_id - 1, overview_column_name]  = key
                except:
                    data = []
                    key = 'not'
                    #data_raw_key_trials[len(data_raw_key_trials)-1] = ['not_' + key]
                else:
                    #Add subject id, speed, trial, and treadmill (1) to each datafile
                    data['subject_id'] = np.zeros(len(data)) + subject_id
                    data['trial_id'] = np.zeros(len(data)) + trial_id
                    data['speed'] = np.zeros(len(data)) + speed
                    data['tread or overground'] = np.zeros(len(data)) + 1 # 1 is for treadmill
                
                data_raw_speeds.append(data)
                data_raw_key_speeds.append(key) 
                
            data_raw_key_trials.append(data_raw_key_speeds)
            data_raw_trials.append(data_raw_speeds)
    
        data_treadmill_raw_key.append(data_raw_key_trials)
        data_treadmill_raw.append(data_raw_trials)
        
    return data_treadmill_raw_key, data_treadmill_raw, overview_file
    



def import_overground_data(dir_data_raw, subjects, trials, overview_file, x_debugging):
    """import raw data and add subject id and trial id to each dataset"""
    # creates lists of lists of lists (2 layers)
    # data_overground_raw_key holds the names of each dataset from data_overground_raw
    # first layer: subjects, second layer: trials, third layer: actual data
    data_overground_raw_key = list('0') # index place 0 with 0 to store stubject 1 at index 1
    data_overground_raw = list('0')
    for subject_id in subjects:
        subject_name = 'SENSOR' + str(subject_id).zfill(3)
        dir_data_raw_subject = dir_data_raw + subject_name + '/'
        data_raw_trials = list()
        data_raw_key_trials = list()
        
        for trial_id in trials:
            if trial_id == 1:
                key = subject_name + '_run'
                overview_column_name = 'overground'
            else:
                key = subject_name + '_run' + '_' + str(trial_id)
                overview_column_name = 'overground_' + str(trial_id)
        
            
            # because not all subjects do have all runs, we try it, if there is an error, we add an empty list to the data and modify the key with a "not" 
            try:
                if x_debugging:
                    key = key + '_x'
                data = pd.read_csv(dir_data_raw_subject + key + '.csv')
                overview_file.loc[subject_id - 1, overview_column_name]  = key
            except:
                data = []
                key = 'not'
            else:
                #Add subject id, speed, and trial to each datafile
                data['subject_id'] = np.zeros(len(data)) + subject_id
                data['trial_id'] = np.zeros(len(data)) + trial_id
                data['tread or overground'] = np.zeros(len(data)) + 0 # 0 is for overground
                
            data_raw_key_trials.append(key)      
            data_raw_trials.append(data)
    
        data_overground_raw_key.append(data_raw_key_trials)
        data_overground_raw.append(data_raw_trials)
        
    return data_overground_raw_key, data_overground_raw, overview_file

def clean_treadmill_data(data_treadmill_raw, data_treadmill_raw_key, plot_flag_treadmill):
    """cut out unusable data from treadmill running""" 
    for i in range(1,len(data_treadmill_raw)):
        for z in range(0,len(data_treadmill_raw[i])):
            for a in range(0,len(data_treadmill_raw[i][z])):
                ## only run if there is data inside 
                try:
                    # function to find start and end of the cleaned/cut data
                    data_start, data_end = cut_treadmill_data_alternative(data_treadmill_raw[i][z][a].left_ay)
                    # add True for usble data and False to none usable
                    #data_start = 19000
                    #data_end = data_start + 25000
                    usabledata = np.zeros(len(data_treadmill_raw[i][z][a]), dtype=bool)
                    usabledata[data_start:data_end] = True
                    # add to dataframe
                    data_treadmill_raw[i][z][a]['usabledata'] = usabledata
                    
                    if plot_flag_treadmill:
                        plt.figure()
                        plt.title(data_treadmill_raw_key[i][z][a])
                        raw_left_ax = plt.plot(data_treadmill_raw[i][z][a].left_ax, label = 'raw_left_ax', color = 'b')
                        raw_left_ay = plt.plot(data_treadmill_raw[i][z][a].left_ay, label = 'raw_left_ay', color = 'b')
                        raw_left_az = plt.plot(data_treadmill_raw[i][z][a].left_az, label = 'raw_left_az', color = 'b')
                        new_left_ax = plt.plot(data_treadmill_raw[i][z][a].left_ax[(data_treadmill_raw[i][z][a]['usabledata'] == True)], label = 'raw_left_ax', color = 'r')
                        new_left_ay = plt.plot(data_treadmill_raw[i][z][a].left_ay[(data_treadmill_raw[i][z][a]['usabledata'] == True)], label = 'raw_left_ay', color = 'r')
                        new_left_az = plt.plot(data_treadmill_raw[i][z][a].left_az[(data_treadmill_raw[i][z][a]['usabledata'] == True)], label = 'raw_left_az', color = 'r')
                        plt.legend()
                except:
                    print(data_treadmill_raw_key[i][z][a] + ' did not work.')
                    
    return data_treadmill_raw


def clean_overground_data(data_overground_raw, data_overground_raw_key, plot_flag_overground):
    for i in range(1,len(data_overground_raw)):
        for z in range(0,len(data_overground_raw[i])):
            """add some comments to this code man"""
            
            ## only run if there is data inside    
            try:
                # function to find start and end of the cleaned/cut data
                data_start_1, data_end_1, data_start_2, data_end_2      \
                    = cut_overground_data(data_overground_raw[i][z].left_ay)
                # add True for usable data and False to none usable
                usabledata = np.zeros(len(data_overground_raw[i][z]), dtype=bool)
                usabledata[data_start_1:data_end_1] = True
                usabledata[data_start_2:data_end_2] = True
                # add to dataframe
                data_overground_raw[i][z]['usabledata'] = usabledata
                
                if plot_flag_overground:
                     plt.figure()
                     plt.title(data_overground_raw_key[i][z])
                     raw_left_ax = plt.plot(data_overground_raw[i][z].left_ax, label = 'raw_left_ax', color = 'b')
                     raw_left_ay = plt.plot(data_overground_raw[i][z].left_ay, label = 'raw_left_ay', color = 'b')
                     raw_left_az = plt.plot(data_overground_raw[i][z].left_az, label = 'raw_left_az', color = 'b')
                     new_left_ax = plt.plot(data_overground_raw[i][z].left_ax[(data_overground_raw[i][z]['usabledata'] == True)], label = 'new_left_ax', color = 'r')
                     new_left_ay = plt.plot(data_overground_raw[i][z].left_ay[(data_overground_raw[i][z]['usabledata'] == True)], label = 'new_left_ay', color = 'r')
                     new_left_az = plt.plot(data_overground_raw[i][z].left_az[(data_overground_raw[i][z]['usabledata'] == True)], label = 'new_left_az', color = 'r')
                     #plt.legend()
             
            except:
                print(data_overground_raw_key[i][z] + ' did not work.')
                
    return data_overground_raw

def cut_treadmill_data_alternative(acc):
    """generalized code to cut out the beginning and end of treadmill running"""   
    # calculate mean of each absolute acceleration
    abs_mean = np.mean(abs(acc))
    
    # create heavily filter data and shifted filtered data
    b, a = sps.butter(4, 0.002, 'low')
    filtered_abs_acc = sps.filtfilt(b, a, abs(acc))
    
    
    max_filtered_abs_acc = np.max(filtered_abs_acc)
    min_filtered_abs_acc = np.min(filtered_abs_acc)
    mean_max_min = np.mean([max_filtered_abs_acc, min_filtered_abs_acc])
    diff_filtered_abs_acc = np.diff(filtered_abs_acc)
    location_max_diff = np.argmax(diff_filtered_abs_acc)
    
    #plt.figure()
    #plt.plot(dif)
    
    
    p = 0
    threshold = mean_max_min
    window = 3000 # how far ahead should it look for the change in acceleration?
    try:
        for i in range(0,len(filtered_abs_acc)):
            ## find start point and end point of constant running
            if filtered_abs_acc[i+window] - filtered_abs_acc[i] > threshold and filtered_abs_acc[i+window + 2000] - filtered_abs_acc[i] > threshold and p == 0:
                flat_1 = i + 5000 # because it is treadmill running, so constant speed occurs at the same time for each subject
                p = 1
            if filtered_abs_acc[i] - filtered_abs_acc[i+window] > threshold and p == 1:
                drop_1 = i # start of the drop
                p = 2
            if filtered_abs_acc[i] - filtered_abs_acc[i+window] < threshold/4 and p == 2:
                drop_2 = i - 500 # end of the drop minus 5000
                break
            
    except:
        ## if it does not work we will do an alternative analysis
        diff_filtered_abs_acc = np.diff(filtered_abs_acc)
        # find max in first half of data (should be onset of run)
        location_max_diff = np.argmax(diff_filtered_abs_acc[0:(round(len(diff_filtered_abs_acc)/2))])
        flat_1 = location_max_diff + 3000
        location_min_diff = np.argmin(diff_filtered_abs_acc[round(len(diff_filtered_abs_acc)/2):len(diff_filtered_abs_acc)]) + round(len(diff_filtered_abs_acc)/2)
        drop_2 = location_min_diff - 500
        
        
        
    ## now find the data that is right in the middle between flat_1 and drop_1 and 32,000 datapoints long. 32,000 because most runs have the same length
    usable_data_length = 25000
    for i in range(flat_1,drop_2):
        empty_window_beginning = i - flat_1 # space between flat_1 and the start of our usable data
        empty_window_end = drop_2 - i - usable_data_length
        if empty_window_beginning - empty_window_end <= 1 and empty_window_beginning - empty_window_end >= -1:
            start = i
            end = i + usable_data_length
            break
        
    return start, end

def cut_overground_data(acc):
    """generalized code to cut out the beginning and end of treadmill running"""   
    # low pass filter over absolute acceleration
    b, a = sps.butter(4, 0.002, 'low')
    filtered_abs_acc = sps.filtfilt(b, a, abs(acc))
    
    ## find when low pass filtered starts rising and falling
    
    # mean_max_min will be funament for threshold
    max_filtered_abs_acc = np.max(filtered_abs_acc)
    min_filtered_abs_acc = np.min(filtered_abs_acc)
    mean_max_min = np.mean([max_filtered_abs_acc, min_filtered_abs_acc])
    
    
    
    counter = 0
    threshold = mean_max_min/2 
    window = 500 # how far ahead should it look for the change in acceleration?
    try:
        
        for i in range(0,len(filtered_abs_acc)):
            if filtered_abs_acc[i+window] - filtered_abs_acc[i] > threshold and counter == 0:
                rise_1 = i
                counter = 1
            if filtered_abs_acc[i+window] - filtered_abs_acc[i] < threshold and counter == 1:
                flat_1 = i + 1000
                counter = 2
            if  filtered_abs_acc[i] - filtered_abs_acc[i+window] > threshold and counter == 2:
                drop_1 = i
                counter = 3
            if filtered_abs_acc[i+window] - filtered_abs_acc[i] > threshold and counter == 3:
                rise_2 = i
                counter = 4
            if filtered_abs_acc[i+window] - filtered_abs_acc[i] < threshold and counter == 4:
                flat_2 = i + 1000
                counter = 5
            if filtered_abs_acc[i] - filtered_abs_acc[i+window] > threshold and counter == 5:
                drop_2 = i - 500
                break
        print('normal method')
    
    except:
        #pdb.set_trace() 
        print('alternative method' )
        ## if it does not work we will do an alternative analysis
        diff_filtered_abs_acc = np.diff(filtered_abs_acc)
        # find max in first third of data (should be onset of first direction run)
        location_max_diff = np.argmax(diff_filtered_abs_acc[0:(round(len(diff_filtered_abs_acc)/3))])
        flat_1 = location_max_diff + 1000
        
        # find min in middle third (should be end of first direction run)
        location_min_diff = np.argmin(diff_filtered_abs_acc[round(len(diff_filtered_abs_acc)/4):round(len(diff_filtered_abs_acc)/4*3)]) + round(len(diff_filtered_abs_acc)/4)
        drop_1 = location_min_diff - 800
        
        # find max in middle third (should be start of second direction run)
        location_max_diff = np.argmax(diff_filtered_abs_acc[round(len(diff_filtered_abs_acc)/4):round(len(diff_filtered_abs_acc)/4*3)]) + round(len(diff_filtered_abs_acc)/4)
        flat_2 = location_max_diff + 1000
        
        # find min in last third of data (should be end of second direction run)
        location_min_diff = np.argmin(diff_filtered_abs_acc[round(len(diff_filtered_abs_acc)/3)*2:len(diff_filtered_abs_acc)]) + round(len(diff_filtered_abs_acc)/3*2)
        drop_2 = location_min_diff - 1500
        
        
        
            
            
    return flat_1, drop_1, flat_2, drop_2
        
    

def cut_treadmill_data(acc):
    """generalized code to cut out the beginning and end of treadmill running"""   
    # calculate mean of each absolute acceleration
    abs_mean = np.mean(abs(acc))   
    
    # create heavily filter data and shifted filtered data
    b, a = sps.butter(4, 0.002, 'low')
    filtered_abs_acc = sps.filtfilt(b, a, abs(acc))
    
    
    ### find point where 1) filtered data crosses mean 2) its own shifted filtered data crosses the filtered data
    
    ## 1
    over_mean = abs(filtered_abs_acc) > abs_mean
    true_indizes = [i for i, value in enumerate(over_mean) if value == True]
    mean_upcrossing = true_indizes[0]
    mean_downcrossing = true_indizes[len(true_indizes)-1]
    first_cut_filtered = filtered_abs_acc[mean_upcrossing:mean_downcrossing]
    
    ## 2
    first_cut_filtered_shifted = first_cut_filtered[1000:len(first_cut_filtered)]
    shifted_lower = first_cut_filtered_shifted < first_cut_filtered[0:len(first_cut_filtered)-1000]
    
    # first time the shifted filtered goes over the original filtered
    true_indizes = [i for i, value in enumerate(shifted_lower) if value == True]
    shifted_upcrossing = true_indizes[0]
    
    # last time the shifted filtered is over the original filtered
    false_indizes = [i for i, value in enumerate(shifted_lower) if value == False]
    shifted_downcrossing = false_indizes[len(false_indizes)-1]
    
    # find the indizes of the original data
    data_start = mean_upcrossing + shifted_upcrossing
    data_end = mean_upcrossing + shifted_downcrossing
    
    return data_start, data_end
    




def create_example(example):
    def bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


    def float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    
    def int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def float_feature_list(value):
        """Returns a list of float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    feature = {
        "left_ax": float_feature_list([x['left_ax'] for x in example['samples']]),
        "left_ay": float_feature_list([x['left_ay'] for x in example['samples']]),
        "left_az": float_feature_list([x['left_az'] for x in example['samples']]),
        "left_gx": float_feature_list([x['left_gx'] for x in example['samples']]),
        "left_gy": float_feature_list([x['left_gy'] for x in example['samples']]),
        "left_gz": float_feature_list([x['left_gz'] for x in example['samples']]),
        "right_ax": float_feature_list([x['right_ax'] for x in example['samples']]),
        "right_ay": float_feature_list([x['right_ay'] for x in example['samples']]),
        "right_az": float_feature_list([x['right_az'] for x in example['samples']]),
        "right_gx": float_feature_list([x['right_gx'] for x in example['samples']]),
        "right_gy": float_feature_list([x['right_gy'] for x in example['samples']]),
        "right_gz": float_feature_list([x['right_gz'] for x in example['samples']]),
        "feature_matrix": float_feature_list(
            pd.DataFrame.from_records([x for x in example['samples']]).values.reshape(-1)
        ),
        "bin_label": int64_feature(example["bin_label"]),
        "seconds_10k": int64_feature(example["seconds_10k"]),
        "subject_id": int64_feature(example["subject_id"]),
        "tread_or_overground": int64_feature(example["tread_or_overground"]),
        "speed": int64_feature(example["speed"]),
        "speed_onehot": float_feature_list(tf.keras.utils.to_categorical(example["speed"], 
                                                                        num_classes=4)
                                                                       ),
        "subject_id_onehot": float_feature_list(tf.keras.utils.to_categorical(example["subject_id"]-1, 
                                                                        num_classes=188)
                                                                       )
                                         
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_example_spectrogram(example):
    def bytes_feature_spectrogram(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))): # if value ist tensor
            value = value.numpy() # get value of tensor
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))
    
    def float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    
    def int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def float_feature_list(value):
        """Returns a list of float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    def serialize_array(array):
        array = tf.io.serialize_tensor(array)
        return array
    
    feature = {
        'height' : int64_feature(example["height"]),
        'width' : int64_feature(example["width"]),
        'depth' : int64_feature(example["depth"]),
        'spectrogram_image' : bytes_feature_spectrogram(serialize_array(example["spectrogram_image"])),
        "filename": bytes_feature(example["filename"]),
        "fullpath": bytes_feature(example["fullpath"]),
        "score_10k": int64_feature(example["score_10k"]),
        "subject_id": int64_feature(example["subject_id"]),
        "tread_or_overground": int64_feature(example["tread_or_overground"])
        
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_single_image_spectrogram(spectrogram, score_10k, seconds_10k, subject_id, bin_label, speed_label):
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))): # if value ist tensor
            value = value.numpy() # get value of tensor
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
      """Returns a floast_list from a float / double."""
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    def _int64_feature(value):
      """Returns an int64_list from a bool / enum / int / uint."""
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def serialize_array(array):
      array = tf.io.serialize_tensor(array)
      return array

    #define the dictionary -- the structure -- of our single example
    data = {
          'height' : _int64_feature(spectrogram.shape[0]),
          'width' : _int64_feature(spectrogram.shape[1]),
          'depth' : _int64_feature(spectrogram.shape[2]),
          'spectrogram_image' : _bytes_feature(serialize_array(spectrogram)),
          'score_10k' : _int64_feature(score_10k),
          'seconds_10k': _int64_feature(seconds_10k),
          'subject_id': _int64_feature(subject_id),
          'bin_label': _int64_feature(bin_label),
          'speed_label': _int64_feature(speed_label)
      }
    
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))
  
    return out


def write_spectrograms_to_tfr(images, labels_score, labels_seconds, filedir, subject_id_array, bin_array, speed_array):
    filedir= filedir + ".tfrecords"
    writer = tf.io.TFRecordWriter(filedir) #create a writer that'll store our data to disk
    count = 0
  
    for index in range(len(images)):
      
      #get the data we want to write
      current_image = images[index] 
      current_label_score = labels_score[index]
      current_label_seconds = labels_seconds[index]
      subject_id = subject_id_array[index]
      bin_label = bin_array[index]
      speed_label = speed_array[index]
   
      
      out = parse_single_image_spectrogram(spectrogram=current_image, score_10k=current_label_score, seconds_10k = current_label_seconds, subject_id = subject_id, bin_label = bin_label, speed_label = speed_label)
      writer.write(out.SerializeToString())
      count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count
