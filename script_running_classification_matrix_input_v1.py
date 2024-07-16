#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:31:23 2022

@author: patrickmayerhofer
script_running_classification_v5

loads the tfrecords, and uses them to do optimize a neural network

Based on:
https://www.kaggle.com/code/danmaccagnola/activity-recognition-data-w-tfrecords/notebook

and for model this could help. consider using a cnn-lstm model
https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
"""
import keras
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
import tensorflow as tf
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve, PrecisionRecallDisplay
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D,LSTM,Activation
import random
import os
from statistics import mean


"""some variables"""
n_timesteps = 5000 # predefined, when we saved tfrecords in "create_TFRecords5"
n_features = 3 # same here. 3 accelerations, 3 angular velocities, 2 feet
#examples_per_file = 64 # same here, these are the nr of files in one tfrecord
epochs = 30
subjects = [110,111,112,113,114,115,116,118,119,120]
val_split = 0.2
test_split = 0.2 #
batch_size = 32
examples_per_file = 27 # the number of examples that are in one TFRecord
# rest will be test set

"""directories"""
dir_root = '/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Data/Prepared/'
dir_tfr = dir_root + "tfrecords/"
dir_tfr_treadmill = dir_tfr + 'Treadmill/'


"""get all directories"""
filenames = list()
for subject in subjects:
    sensor = "SENSOR" + "{:03d}".format(subject)
    dir_subject = dir_tfr_treadmill + "SENSOR" + "{:03d}".format(subject) + "/"
    filenames_temporary = tf.io.gfile.glob(f"{dir_subject}*.tfrec")
    filenames.append(filenames_temporary)
    
#shuffle subject list
random.shuffle(filenames)

"""divide in train and test set directories"""
test_filenames_temp = filenames[0:int(len(filenames)*test_split)]
val_filenames_temp = filenames[int(len(filenames)*test_split):int(len(filenames)*test_split)+int(len(filenames)*val_split)]
train_filenames_temp = filenames[int(len(filenames)*test_split)+int(len(filenames)*val_split):len(filenames)]

# flatten lists and shuffle
train_filenames = [item for sublist in train_filenames_temp for item in sublist]
test_filenames = [item for sublist in test_filenames_temp for item in sublist]
val_filenames = [item for sublist in val_filenames_temp for item in sublist]

# shuffle again for each individual directory
random.shuffle(train_filenames)
random.shuffle(test_filenames)
random.shuffle(val_filenames)

print(f"Train: {len(train_filenames)}")
print(f"Validation: {len(val_filenames)}")
print(f"Test: {len(test_filenames)}")

"""
for batch in tf.data.TFRecordDataset(filenames):
    print(batch)
    break
"""


"""parse serialized data function"""
def parse_tfrecord_fn(example):
    
    feature_description = {
        "filename": tf.io.FixedLenFeature([], tf.string),
        "fullpath": tf.io.FixedLenFeature([], tf.string),
        "score_10k": tf.io.FixedLenFeature([], tf.int64),
        "left_ax": tf.io.VarLenFeature(tf.float32),
        "left_ay": tf.io.VarLenFeature(tf.float32),
        "left_az": tf.io.VarLenFeature(tf.float32),
        "feature_matrix": tf.io.VarLenFeature(tf.float32),
        "subject_id": tf.io.FixedLenFeature([], tf.int64),
        'tread_or_overground': tf.io.VarLenFeature(tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["feature_matrix"] = tf.reshape(tf.sparse.to_dense(example["feature_matrix"]), (n_timesteps, n_features, 1))
    return example


"""
for batch in tf.data.TFRecordDataset(filenames).map(parse_tfrecord_fn):
    print(batch)
    break
"""

"""prepare input and output for model"""
def prepare_sample(features):
    #image = tf.image.resize(features["image"], size=(224, 224))
    #return image, features["category_id"]
    input_data = features["feature_matrix"]
    output_data = features['score_10k']
    
    return input_data, output_data

"""
(tf.data.TFRecordDataset(filenames)
 .map(parse_tfrecord_fn)
 .map(prepare_sample)
)
"""

"""fetch the data"""
AUTOTUNE = tf.data.AUTOTUNE


for batch in tf.data.TFRecordDataset(train_filenames).map(parse_tfrecord_fn):
    tens = batch
    print(batch)
    break

def get_dataset(filenames, batch_size):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=AUTOTUNE)
        .shuffle(batch_size * 10)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset


get_dataset(train_filenames, batch_size)


"""create neural network """
model = tf.keras.Sequential()
model.add(layers.Conv2D(40,5, activation = "relu", input_shape = (30,6,1)))
model.output_shape
model = tf.keras.Sequential([
    layers.Conv2D(16, 3, activation = "relu", input_shape = (30,3,1)),
    #layers.MaxPool2D(2),
    layers.Reshape((28,16,1)),
    layers.Conv2D(32, 3, activation = "relu"),
    layers.MaxPool2D(2),
    layers.Flatten(),
    layers.Dense(16, activation = "relu"),
    layers.Dense(1, activation = "relu") # softmax va bene per i multiclass, altrimenti uso sigmoid
])
model.summary()
model.compile(loss='mean_absolute_error',
       optimizer='adam', metrics=['mean_absolute_error'])

"""
model = Sequential()
model.add(Dense(128,activation = 'relu',
               input_shape=(n_timesteps, n_features,)))  # returns a sequence of vectors of dimension 32
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation='relu'))
model.summary()
model.compile(loss='mean_absolute_error',
       optimizer='adam', metrics=['mean_absolute_error'])
"""
"""
model = Sequential()
model.add(LSTM(32,activation = 'relu',
               input_shape=(n_timesteps, n_features), return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, activation = 'relu'))
model.add(Dense(1, activation='relu'))
model.summary()
model.compile(loss='mean_absolute_error',
       optimizer='adam', metrics=['mean_absolute_error'])
"""

get_dataset(train_filenames, batch_size)

steps_per_epoch = np.int(np.ceil(examples_per_file*len(train_filenames)/batch_size))
validation_steps = np.int(np.ceil(examples_per_file*len(val_filenames)/batch_size))
steps = np.int(np.ceil(examples_per_file*len(test_filenames)/batch_size))
print("steps_per_epoch = ", steps_per_epoch)
print("validation_steps = ", validation_steps)
print("steps = ", steps)


train_dataset = get_dataset(train_filenames, batch_size)
val_dataset = get_dataset(val_filenames, batch_size)
test_dataset = get_dataset(test_filenames, batch_size)


model.fit(train_dataset,
          validation_data = val_dataset, 
          #steps_per_epoch = steps_per_epoch,
          #validation_steps = validation_steps, 
          epochs = epochs
         )


#get_dataset(filenames, batch_size)



"""calculate accuracy in different ways"""
evaluated_accuracy = model.evaluate(test_dataset)
evaluated_accuracy
pred_values = model.predict(test_dataset)

steps_to_take = len(test_filenames)

pred_values_list = []
pred_list = []
true_list = []
"""
for x, y in test_dataset.take(steps_to_take):
    
    pred_value = model.predict(x)
    pred_values_list = pred_values_list + list(pred_value)
"""


real_data = list(test_dataset.as_numpy_iterator())

true_values_temp = list()
for i in range(len(real_data)):
    true_values_temp.append(real_data[i][1])
    
#flatten list if not flat
true_values = [item for sublist in true_values_temp for item in sublist]


#manually calculated accuracy
#pred_values_only_last = list(pred_values[:, len(pred_values[0,:,0])-1, 0])
error = list()
for i in range(0, len(pred_values)):
    error.append(abs(pred_values[i]-true_values[i]))

mean_abs_error = sum(error)/len(error)
    

    
