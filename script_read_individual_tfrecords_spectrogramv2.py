#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 08:38:42 2022

@author: patrickmayerhofer
"""
import functions_classification_general as fcg
import matplotlib.pyplot as plt

"""some variables"""
which_spectrograms = '10k_125_bins2_onespeed'
batch_size = 32

"""directories"""
dir_root = '/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Data/Prepared/'
dir_tfr_spectrogram = dir_root + "tfrecords/" + which_spectrograms +  "/Treadmill/"
dir_subject = dir_tfr_spectrogram + "SENSOR005.tfrecords"


dataset = fcg.get_dataset(dir_subject, batch_size)

for sample in dataset.take(1):
  print(sample[0].shape)
  print(sample[1].shape)
  print(sample)
  one_layer = sample[0][0,:,:,0]
  dataset.take(1)
  break

plt.figure()
plt.imshow(one_layer, cmap='hot', interpolation='nearest')
plt.show()  

plt.figure()
plt.pcolormesh(one_layer, shading='gouraud')
plt.ylabel('Frequency [db]')
plt.xlabel('Time [sec]')

