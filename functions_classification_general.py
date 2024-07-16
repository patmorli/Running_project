#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:53:21 2022

@author: patrick
"""

import tensorflow as tf
import keras
import numpy as np
AUTOTUNE = tf.data.AUTOTUNE

"""parse serialized data function"""
def parse_tfr_element_bins(element):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
  'height': tf.io.FixedLenFeature([], tf.int64),
  'width':tf.io.FixedLenFeature([], tf.int64),
  'depth':tf.io.FixedLenFeature([], tf.int64),
  'spectrogram_image' : tf.io.FixedLenFeature([], tf.string),
  'score_10k': tf.io.FixedLenFeature([], tf.int64),
  'seconds_10k': tf.io.FixedLenFeature([], tf.int64),
  'subject_id': tf.io.FixedLenFeature([], tf.int64),
  'bin_label': tf.io.FixedLenFeature([], tf.int64),
  'speed_label': tf.io.FixedLenFeature([], tf.int64)
}


  content = tf.io.parse_single_example(element, data)
  
  height = content['height']
  width = content['width']
  depth = content['depth']
  score_10k = content['score_10k']
  seconds_10k = content['seconds_10k']
  spectrogram_image = content['spectrogram_image']
  bin_label = content['bin_label']
  bin_label = tf.one_hot(bin_label, depth = 2, dtype = 'int64')
  subject_id = content['subject_id']-1
  subject_id = tf.one_hot(subject_id, depth = 10, dtype = 'int64')
  speed_label = content['speed_label']-1
  speed_label = tf.one_hot(speed_label, depth = 3, dtype = 'int64')
  
  
  #get our 'feature'-- our image -- and reshape it appropriately
  feature = tf.io.parse_tensor(spectrogram_image, out_type=tf.double)
  feature = tf.reshape(feature, shape=[height,width,depth])
  return (feature, speed_label)

def parse_tfr_element_cont(element):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width':tf.io.FixedLenFeature([], tf.int64),
      'depth':tf.io.FixedLenFeature([], tf.int64),
      'spectrogram_image' : tf.io.FixedLenFeature([], tf.string),
      'score_10k': tf.io.FixedLenFeature([], tf.int64),
      'seconds_10k': tf.io.FixedLenFeature([], tf.int64),
      'subject_id': tf.io.FixedLenFeature([], tf.int64),
      'bin_label': tf.io.FixedLenFeature([], tf.int64),
      'speed_label': tf.io.FixedLenFeature([], tf.int64)
    }

    
  content = tf.io.parse_single_example(element, data)
  
  height = content['height']
  width = content['width']
  depth = content['depth']
  score_10k = content['score_10k']
  seconds_10k = content['seconds_10k']
  spectrogram_image = content['spectrogram_image']
  bin_label = content['bin_label']
  bin_label = tf.one_hot(bin_label, depth = 2, dtype = 'int64')
  

  
  
  
  #get our 'feature'-- our image -- and reshape it appropriately
  feature = tf.io.parse_tensor(spectrogram_image, out_type=tf.double)
  feature = tf.reshape(feature, shape=[height,width,depth])
  #feature = feature[3]
  return (feature, seconds_10k)


def get_dataset_bins(filenames, batch_size):
  #create the dataset
  dataset = (
          tf.data.TFRecordDataset(filenames, num_parallel_reads=None)
          .map(parse_tfr_element_bins)
          .shuffle(batch_size*10)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE)
          )
    
  return dataset

def get_dataset_bins_unshuffled(filenames, batch_size):
  #create the dataset
  dataset = (
          tf.data.TFRecordDataset(filenames, num_parallel_reads=None)
          .map(parse_tfr_element_bins)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE)
          )
    
  return dataset

def get_dataset_cont(filenames, batch_size):
  #create the dataset
  dataset = (
          tf.data.TFRecordDataset(filenames, num_parallel_reads=None)
          .map(parse_tfr_element_cont)
          .shuffle(batch_size*10)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE)
          )
    
  return dataset

def get_dataset_cont_unshuffled(filenames, batch_size):
  #create the dataset
  dataset = (
          tf.data.TFRecordDataset(filenames, num_parallel_reads=None)
          .map(parse_tfr_element_cont)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE)
          )
    
  return dataset


def parse_tfrecord_rnn(example):
    window_length = 5000
    feature_description = {
        "left_ax": tf.io.VarLenFeature(tf.float32),
        "left_ay": tf.io.VarLenFeature(tf.float32),
        "left_az": tf.io.VarLenFeature(tf.float32),
        "left_gx": tf.io.VarLenFeature(tf.float32),
        "left_gy": tf.io.VarLenFeature(tf.float32),
        "left_gz": tf.io.VarLenFeature(tf.float32),
        "right_ax": tf.io.VarLenFeature(tf.float32),
        "right_ay": tf.io.VarLenFeature(tf.float32),
        "right_az": tf.io.VarLenFeature(tf.float32),
        "right_gx": tf.io.VarLenFeature(tf.float32),
        "right_gy": tf.io.VarLenFeature(tf.float32),
        "right_gz": tf.io.VarLenFeature(tf.float32),
        "feature_matrix": tf.io.VarLenFeature(tf.float32),
        "bin_label": tf.io.FixedLenFeature([], tf.int64),
        "seconds_10k": tf.io.FixedLenFeature([], tf.int64),
        "subject_id": tf.io.FixedLenFeature([], tf.int64),
        'tread_or_overground': tf.io.VarLenFeature(tf.int64),
        "speed": tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["left_ax"] = tf.sparse.to_dense(example["left_ax"])
    example["left_ay"] = tf.sparse.to_dense(example["left_ay"])
    example["left_az"] = tf.sparse.to_dense(example["left_az"])
    example["left_gx"] = tf.sparse.to_dense(example["left_gx"])
    example["left_gy"] = tf.sparse.to_dense(example["left_gy"])
    example["left_gz"] = tf.sparse.to_dense(example["left_gz"])
    example["right_ax"] = tf.sparse.to_dense(example["right_ax"])
    example["right_ay"] = tf.sparse.to_dense(example["right_ay"])
    example["right_az"] = tf.sparse.to_dense(example["right_az"])
    example["right_gx"] = tf.sparse.to_dense(example["right_gx"])
    example["right_gy"] = tf.sparse.to_dense(example["right_gy"])
    example["right_gz"] = tf.sparse.to_dense(example["right_gz"])
    example["feature_matrix"] = tf.reshape(tf.sparse.to_dense(example["feature_matrix"]), (window_length, 12, 1))
    example['subject_id'] = tf.one_hot(example['subject_id']-1, depth = 2, dtype = 'int64')
    example["speed"] = tf.one_hot(example['speed']-1, depth = 3, dtype = 'int64')
    return example



def prepare_sample_rnn(features):
    #image = tf.image.resize(features["image"], size=(224, 224))
    #return image, features["category_id"]
    input_data = features["left_ay"]
    output_data = features['speed']
    
    return input_data, output_data

def prepare_sample_rnn_cont(features):
    #image = tf.image.resize(features["image"], size=(224, 224))
    #return image, features["category_id"]
    input_data = features["feature_matrix"]
    output_data = features['seconds_10k']
    
    return input_data, output_data

def get_dataset_rnn(filenames, batch_size):
    
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_rnn, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample_rnn, num_parallel_calls=AUTOTUNE)
        .shuffle(batch_size * 10)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset

def get_dataset_rnn_unshuffled(filenames, batch_size):
    
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_rnn, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample_rnn, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset

def get_dataset_rnn_cont(filenames, batch_size):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_rnn, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample_rnn, num_parallel_calls=AUTOTUNE)
        .shuffle(batch_size * 10)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset

def get_predictions_true_manually(dataset, my_model, steps_to_take):
    pred_list_val = []
    true_list_val = []
    for x, y in dataset.take(steps_to_take):
        
        pred_values_val = my_model.predict(x)
        pred_list_val = pred_list_val + list(pred_values_val)
        true_list_val = true_list_val + list(y.numpy())
        pred_list_val_argmax = np.argmax(pred_list_val, axis=1)
        
    return true_list_val, pred_list_val, x


"To test some of the functionality if there is a problem"

"""
for batch in tf.data.TFRecordDataset(filenames):
    print(batch)
    break
"""

"""
for batch in tf.data.TFRecordDataset(filenames).map(parse_tfr_element):
    print(batch)
    break
"""


"""prepare input and output for model"""

"""
def prepare_sample(features):
    input_data = features["feature_matrix"]
    output_data = features['score_10k']
    
    return input_data, output_data
"""
"""
(tf.data.TFRecordDataset(filenames)
 .map(parse_tfr_element)
 .map(prepare_sample)
)
"""
