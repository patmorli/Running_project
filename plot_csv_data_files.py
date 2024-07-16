#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:40:51 2022

@author: patrickmayerhofer

plot_csv_data_files
"""

import pandas as pd
import matplotlib.pyplot as plt


data_id = '184_3.0'

tread_or_over = 'Treadmill'

# my helpful directories
dir_root = '/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Data/Prepared/'
dir_data = dir_root +  tread_or_over + '/SENSOR' + data_id + '.csv'

#import
data = pd.read_csv(dir_data)

#plot
plt.figure()
raw_left_ax = plt.plot(data.left_ax, label = 'raw_left_ax', color = 'b')
raw_left_ay = plt.plot(data.left_ay, label = 'raw_left_ay', color = 'b')
raw_left_az = plt.plot(data.left_az, label = 'raw_left_az', color = 'b')
new_left_ax = plt.plot(data.left_ax[(data['usabledata'] == True)], label = 'raw_left_ax', color = 'r')
new_left_ay = plt.plot(data.left_ay[(data['usabledata'] == True)], label = 'raw_left_ay', color = 'r')
new_left_az = plt.plot(data.left_az[(data['usabledata'] == True)], label = 'raw_left_az', color = 'r')
plt.legend()



