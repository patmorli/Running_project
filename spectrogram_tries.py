#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:38:42 2022

@author: patrick

spectrogram_tries
"""

import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from scipy import signal
from scipy.fft import fftshift
from matplotlib import mlab
import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import collections

fs = 500
nperseg=250
noverlap=0

data1_name = 'sub55_gx_2_5'
data2_name = 'sub55_gx_3_0'


data1 = pd.read_csv(data1_name + '.csv')
data1 = data1.left_gx
numpy_data1 = data1.to_numpy()

data2 = pd.read_csv(data2_name + '.csv')
data2 = data2.left_gx
numpy_data2 = data2.to_numpy()

"""Graph 1 - decibel"""
f1, t1, Sxx1 = signal.spectrogram(data1, fs, nperseg=nperseg, noverlap=noverlap)
Sxx_to_db1 = librosa.power_to_db(Sxx1, ref=np.max)

plt.figure(1)
plt.pcolormesh(Sxx_to_db1, shading='gouraud')
#plt.pcolormesh(t1, f1, Sxx_to_db1, shading='gouraud')
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.title(data1_name + ' - Linear')

# Plot with Logarithmic Frequency
#fig, ax = plt.subplots()
plt.figure(2)
librosa.display.specshow(Sxx_to_db1, sr=fs, hop_length=nperseg, x_axis='s', y_axis='log')
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.title(data1_name + ' - Log')







f2, t2, Sxx2 = signal.spectrogram(data2, fs, nperseg=nperseg, noverlap=noverlap)
Sxx_to_db2 = librosa.power_to_db(Sxx2, ref=np.max)

#f = f[0:100]
#Sxx = Sxx[0:100,:]

plt.figure(3)
plt.pcolormesh(Sxx_to_db2, shading='gouraud')
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.title(data2_name + ' - Linear')



# Plot with Logarithmic Frequency
#fig, ax = plt.subplots()
plt.figure(4)
librosa.display.specshow(Sxx_to_db2, sr=fs, hop_length=nperseg, x_axis='s', y_axis='log')
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.title(data2_name + ' - Log')



"""Trying the mel scale stuff"""
"""
numpy_data1 = data1.to_numpy()
mel_spect = librosa.feature.melspectrogram(y=numpy_data1, sr=fs, n_fft=nperseg, hop_length=noverlap)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
librosa.display.specshow(mel_spect, y_axis='mel', x_axis='time');
plt.title('Mel Spectrogram');
plt.colorbar(format='%+2.0f dB');

mel_spect = librosa.power_to_db(Sxx, ref=np.max)
"""

