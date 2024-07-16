#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 08:38:42 2022

@author: patrickmayerhofer
"""

import tensorflow as tf

"""some variables"""
n_timesteps = 1000 # predefined, when we saved tfrecords in "create_TFRecords3"
n_features = 12 # same here. 3 accelerations, 3 angular velocities, 2 feet


"""directories"""
dir_root = '/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Data/Prepared/'
dir_tfr = dir_root + "tfrecords/"
dir_subject = dir_tfr + "SENSOR001/"

filenames = tf.io.gfile.glob(f"{dir_subject}*.tfrec")

"""parse serialized data function"""
def parse_tfrecord_fn(example):
    
    feature_description = {
        "filename": tf.io.FixedLenFeature([], tf.string),
        "fullpath": tf.io.FixedLenFeature([], tf.string),
        "score_10k": tf.io.FixedLenFeature([], tf.int64),
        "feature_matrix": tf.io.VarLenFeature(tf.float32),
        "subject_id": tf.io.FixedLenFeature([], tf.int64),
        'tread_or_overground': tf.io.VarLenFeature(tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["feature_matrix"] = tf.reshape(tf.sparse.to_dense(example["feature_matrix"]), (n_timesteps, n_features, 1))
    return example

"""prepare input and output for model"""
def prepare_sample(features):
    #image = tf.image.resize(features["image"], size=(224, 224))
    #return image, features["category_id"]
    input_data = features["feature_matrix"]
    output_data = features['score_10k']
    
    return input_data, output_data

my_data = list()
for batch in tf.data.TFRecordDataset(filenames).map(parse_tfrecord_fn):
    my_data.append(batch)
    print(batch)