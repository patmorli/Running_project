#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:43:10 2022

@author: patrickmayerhofer

organize_IAAF_score

Takes the unorganized excel files from the IAAF scoring table, and gets it in better shape
"""
import pandas as pd
import re
import copy


dir_root = '/Users/patrickmayerhofer/Google Drive/My Drive/Running Plantiga Project/IAAF Scores/'
filename = 'Women_Unorganized.csv'
save_name = 'Women_Organized.csv'


file = pd.read_csv(dir_root + filename, encoding='ISO-8859-1')

"""save important information in these lists"""
points = list()
m5000 = list()
m10000 = list()
k10 = list()
halfmarathon = list()
marathon = list()


# find the header rows
where_points = file.isin(['Points'])



# 5k and 10k
for i in range(0, 1565):
    # find if row contains 'Points'
    header = file.loc[i,:].isin(['Points'])
    # if it does, find Points, 5k, and 10k and store in respective list
    if header.any():
        # points
        index_points = file.loc[i,:][file.loc[i,:] == 'Points'].index[0]
        points.append(file.loc[i+1:i+50,[index_points]].values.flatten().tolist())
        
        # 5k
        index_m5000 = file.loc[i,:][file.loc[i,:] == '5000m'].index[0]
        m5000.append(file.loc[i+1:i+50,[index_m5000]].values.flatten().tolist())
        
        # 10k
        index_m10000 = file.loc[i,:][file.loc[i,:] == '10000m'].index[0]
        m10000.append(file.loc[i+1:i+50,[index_m10000]].values.flatten().tolist())
        

# half marathon and marathon and second 10k
for i in range(1566, len(file)):
    # find if row contains 'Points'
    header = file.loc[i,:].isin(['Points'])
    # if it does, find Points, 5k, and 10k and store in respective list
    if header.any():
        # points
        #index_points = file.loc[i,:][file.loc[i,:] == 'Points'].index[0]
        #points.append(file.loc[i+1:i+50,[index_points]].values.flatten().tolist())
        
        # 10k
        index_k10 = file.loc[i,:][file.loc[i,:] == '10 km'].index[0]
        k10.append(file.loc[i+1:i+50,[index_k10]].values.flatten().tolist())
        
        # half marathon
        index_halfmarathon = file.loc[i,:][file.loc[i,:] == 'HM'].index[0]
        halfmarathon.append(file.loc[i+1:i+50,[index_halfmarathon]].values.flatten().tolist()) 
        
        # marathon - more complicated because sometimes within "Marathon" and sometimes within "Marathon 100 km"
        marathonthere = file.loc[i,:].isin(['Marathon'])
        if file.loc[i,:].isin(['Marathon']).any():
            index_marathon = file.loc[i,:][file.loc[i,:] == 'Marathon'].index[0]
            marathon.append(file.loc[i+1:i+50,[index_marathon]].values.flatten().tolist()) 
        else:
            index_marathon = file.loc[i,:][file.loc[i,:] == 'Marathon 100 km'].index[0]
            marathon.append(file.loc[i+1:i+50,[index_marathon]].values.flatten().tolist()) 
            

"""create new dataframe and save"""
# flatten list and create series
s_points = pd.Series([item for sublist in points for item in sublist])
s_m5000 = pd.Series([item for sublist in m5000 for item in sublist])
s_m10000 = pd.Series([item for sublist in m10000 for item in sublist])
s_k10 = pd.Series([item for sublist in k10 for item in sublist])
s_halfmarathon = pd.Series([item for sublist in halfmarathon for item in sublist])
s_marathon = pd.Series([item for sublist in marathon for item in sublist])

# create dataframe
organized_file = pd.DataFrame({'Points': s_points, '5000m': s_m5000, '10000m': s_m10000, '10 km': s_k10, 'Halfmarathon': s_halfmarathon, 'Marathon': s_marathon})

iaaf_score = copy.deepcopy(organized_file)
for row in range(0, len(iaaf_score)):
    for column in ['5000m', '10000m', '10 km', 'Halfmarathon', 'Marathon']:
        split_time = re.split('[:.]', iaaf_score.loc[row, [column]].to_string(index=False))
        
        ## different formats 
        if column == '5000m' or column == '10000m':
            seconds = float(split_time[0])*60 + float(split_time[1]) + round(float(split_time[2])*0.1)
                    
        if column == '10 km':
            if row < 1259:
                if len(split_time) > 1:
                    seconds = float(split_time[0])*60 + float(split_time[1])
                else: 
                    seconds = float("nan")
            else:
               if len(split_time) > 1:
                    seconds = float(split_time[0])*60*60 + float(split_time[1])*60 + float(split_time[2])
               else: 
                    seconds = float("nan") 
        
        if column == 'Halfmarathon' or column == 'Marathon':
            if len(split_time ) == 2:
                seconds = float(split_time[0])*60 + float(split_time[1])
            else:
                seconds = float(split_time[0])*60*60 + float(split_time[1])*60 + float(split_time[2])
        
        iaaf_score.loc[row, [column]] = seconds

# rename iaaf_score to mirror the seconds
iaaf_score = iaaf_score.rename({'5000m': 'sec_5000m', '10000m': 'sec_10000m', '10 km': 'sec_10 km', 'Halfmarathon': 'sec_Halfmarathon', 'Marathon': 'sec_Marathon',}, axis=1)

# save
organized_file.to_csv(dir_root + save_name, index = False)
iaaf_score.to_csv(dir_root + 'Women_Organized_Seconds.csv')
    
        
  