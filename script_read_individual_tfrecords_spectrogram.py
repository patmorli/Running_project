#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 08:38:42 2022

@author: patrickmayerhofer
"""

import tensorflow as tf
import matplotlib.pyplot as plt

"""some variables"""
n_timesteps = 10000 # predefined, when we saved tfrecords in "create_TFRecords_spectrogram2"
n_features = 12 # same here. 3 accelerations, 3 angular velocities, 2 feet

n_windows = 12 #4 windows per speed 

"""directories"""
dir_root = '/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Data/Prepared/'
dir_tfr_spectrogram = dir_root + "tfrecords/windows_10000_spectrogram/Treadmill/"
dir_subject = dir_tfr_spectrogram + "SENSOR052.tfrecords"


"""parse serialized data function"""
def parse_tfr_element(element):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width':tf.io.FixedLenFeature([], tf.int64),
      'depth':tf.io.FixedLenFeature([], tf.int64),
      'spectrogram_image' : tf.io.FixedLenFeature([], tf.string),
      'score_10k': tf.io.FixedLenFeature([], tf.int64),
      'seconds_10k': tf.io.FixedLenFeature([], tf.int64),
      'subject_id': tf.io.FixedLenFeature([], tf.int64)
    }

    
  content = tf.io.parse_single_example(element, data)
  
  height = content['height']
  width = content['width']
  depth = content['depth']
  score_10k = content['score_10k']
  seconds_10k = content['seconds_10k']
  spectrogram_image = content['spectrogram_image']
  
  
  #get our 'feature'-- our image -- and reshape it appropriately
  feature = tf.io.parse_tensor(spectrogram_image, out_type=tf.double)
  #feature = tf.reshape(feature, shape=[height,width,depth])
  feature = tf.reshape(feature, shape=[502,502,3])
  return (feature, seconds_10k)

"""
for batch in tf.data.TFRecordDataset(filenames).map(parse_tfr_element):
    print(batch)
    break
"""

def get_dataset_small(filename):
  #create the dataset
  dataset = tf.data.TFRecordDataset(filename)

  #pass every single feature through our mapping function
  dataset = dataset.map(
      parse_tfr_element
  )
    
  return dataset


dataset_small = get_dataset_small(dir_subject)

for sample in dataset_small.take(1):
  print(sample[0].shape)
  print(sample[1].shape)
  print(sample)
  one_layer = sample[0][:,:,0]
  dataset_small.take(1)
  break
  
plt.imshow(one_layer, cmap='hot', interpolation='nearest')
plt.show()  
