#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:54:24 2023

@author: patrick

get mean and std of some participant variables
"""

import pandas as pd
import re


# Set the root directory and file name
dir_root = "/Users/patrick/Google Drive/My Drive/Running Plantiga Project/Data/"
file_name = "Subject_questionnaire.csv"
file_name2 = 'Participant_stats.csv'

# Read in the CSV file
df = pd.read_csv(dir_root + file_name)
df2 = pd.read_csv(dir_root + file_name2)

# Define the participants of interest
participant_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,19,20,21,23,24,25,27,28,30,32,33,34,35,36,37,38,40,42,43,45,46,48,49,52,53,54,55,58,60,61,62,63,66,67,68,69,70,72,73,74,77,79,80,82,84,85,87,88,89,90,91,92,93,94,95,96,98,99,100,101,102,104,105,106,107,108,110,111,112,113,114,115,116,118,119,120,122,123,126,127,128,130,131,132,133,135,138,139,142,143,146,147,150,151,154,156,157,158,159,160,161,162,163]
sensor_values = [f"SENSOR{i:03d}" for i in range(1, len(participant_ids)+1)]
participant_to_sensor = dict(zip(participant_ids, [f"SENSOR{i:03d}" for i in participant_ids]))

# Convert list of participant IDs to SENSOR values
participants_of_interest = [participant_to_sensor[pid] for pid in participant_ids]





# Isolate participants of interest
df = df[df["Q1.3"].astype(str).isin(map(str, participant_ids))]
df["Q1.7"] = pd.to_numeric(df["Q1.7"], errors="coerce")

df2 = df2[df2["SUBJECTID"].isin(participants_of_interest)]

'''Descriptive stats with Subject_questionnaire'''

# Calculate descriptive statistics
num_males = df[df["Q1.6"] == "Male"]["Q1.6"].count()
num_females = df[df["Q1.6"] == "Female"]["Q1.6"].count()
num_man = df[df["Q1.4"] == "Man"]["Q1.4"].count()
num_woman = df[df["Q1.4"] == "Woman"]["Q1.4"].count()


mean_age = df["Q1.7"].mean()
std_age = df["Q1.7"].std()



# Extract shoe size and gender information
df["shoe_size"] = df["Q1.9_1"].apply(lambda x: re.findall(r"\d+\.*\d*", x)[0]).astype(float)

# Calculate mean and std shoe size overall
mean_shoe_size = df["shoe_size"].mean()
std_shoe_size = df["shoe_size"].std()

# Calculate mean and std shoe size by gender
mean_shoe_size_female = df[df["Q1.6"] == "Female"]["shoe_size"].mean()
std_shoe_size_female = df[df["Q1.6"] == "Female"]["shoe_size"].std()
mean_shoe_size_male = df[df["Q1.6"] == "Male"]["shoe_size"].mean()
std_shoe_size_male = df[df["Q1.6"] == "Male"]["shoe_size"].std()

'''All descriptive stats with Participant_stats'''
mean_age2 = df2["AGE (AT BASELINE)"].mean()
std_age2 = df2["AGE (AT BASELINE)"].std()
mean_height2 = df2["HEIGHT (M)"].mean()
std_height2 = df2["HEIGHT (M)"].std()
mean_mass2 = df2["MASS (KG)"].mean()
std_mass2 = df2["MASS (KG)"].std()
mean_shoe_size2 = df2["SHOE SIZE"].mean()



