#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:06:15 2022

@author: patrickmayerhofer
"""

"""This version is easiest, but creates multiple features which are harder to use in training"""
import tensorflow as tf
import os
import tempfile
import numpy as np
example_path = "example.tfrecords"
x, y = np.random.random(4), np.random.random(4)

# Write the records to a file.
with tf.io.TFRecordWriter(example_path) as file_writer:
  for i in range(4):
   
    record_bytes = tf.train.Example(features=tf.train.Features(feature={
        "x": tf.train.Feature(float_list=tf.train.FloatList(value=[x[i]])),
        "y": tf.train.Feature(float_list=tf.train.FloatList(value=[y[i]])),
    })).SerializeToString()
    file_writer.write(record_bytes)
    
    
# Read the data back out.
def decode_fn(record_bytes):
  return tf.io.parse_single_example(
      # Data
      record_bytes,

      # Schema
      {"x": tf.io.FixedLenFeature([], dtype=tf.float32),
       "y": tf.io.FixedLenFeature([], dtype=tf.float32)}
  )


for batch in tf.data.TFRecordDataset([example_path]).map(decode_fn):
  print("x = {x:.4f},  y = {y:.4f}".format(**batch))



