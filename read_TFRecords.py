#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:11:38 2022

@author: patrickmayerhofer

This is just to double check.
TFRecordsToNumpy
Does not reconvert to numpy, but you use it to double check
if the TFRecord was saved the right way. 
Careful, they get saved in a random way in NumpyToTFRecord, 
so need to check the random index over there to know which 
one is supposed to be number 1.

tf.compat.v1.enable_eager_execution() needs to be run when starting kernel.

used this to write and read:
    https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
"""

import tensorflow as tf


dir_data = '/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Data/Prepared/tfrecords/Treadmill/Subject007/SENSOR007_2.5_0.tfrecords'

def decode_fn(record_bytes):
  return tf.io.parse_single_example(
      # Data
      record_bytes,

      # Schema
      {"feature0": tf.io.FixedLenFeature([], dtype=tf.float32),
       "feature1": tf.io.FixedLenFeature([], dtype=tf.float32),
       "feature2": tf.io.FixedLenFeature([], dtype=tf.float32),
       "feature3": tf.io.FixedLenFeature([], dtype=tf.float32),
       "feature4": tf.io.FixedLenFeature([], dtype=tf.float32),
       "feature5": tf.io.FixedLenFeature([], dtype=tf.float32),
       "feature6": tf.io.FixedLenFeature([], dtype=tf.float32),
       "feature7": tf.io.FixedLenFeature([], dtype=tf.float32),
       "feature8": tf.io.FixedLenFeature([], dtype=tf.float32),
       "feature9": tf.io.FixedLenFeature([], dtype=tf.float32),
       "feature10": tf.io.FixedLenFeature([], dtype=tf.float32),
       "feature11": tf.io.FixedLenFeature([], dtype=tf.float32),
       "label": tf.io.FixedLenFeature([], dtype=tf.int64)}
  )


"""how do I change the label to an int not to a float here"""
for batch in tf.data.TFRecordDataset([dir_data]).map(decode_fn):
  print("feature0 = {feature0:.4f},  label = {label:d}".format(**batch))
  
  

"""
       
"""  