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
from keras import layers, models
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve, PrecisionRecallDisplay
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D,LSTM,Activation
import random
import os
from statistics import mean
#import lime
#import lime.lime_tabular


"""some variables"""
n_timesteps = 5000 # predefined, when we saved tfrecords in "create_TFRecords5"
n_features = 12 # same here. 3 accelerations, 3 angular velocities, 2 feet
#examples_per_file = 64 # same here, these are the nr of files in one tfrecord
epochs = 10
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
        "filename": tf.io.FixedLenFeature([], tf.string),
        "fullpath": tf.io.FixedLenFeature([], tf.string),
        "score_10k": tf.io.FixedLenFeature([], tf.int64),
        "subject_id": tf.io.FixedLenFeature([], tf.int64),
        'tread_or_overground': tf.io.VarLenFeature(tf.int64)
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
    example["feature_matrix"] = tf.reshape(tf.sparse.to_dense(example["feature_matrix"]), (n_timesteps, n_features, 1))
    return example


"""
for batch in tf.data.TFRecordDataset(filenames).map(parse_tfrecord_fn):
    print(batch)
    break
"""

"""prepare input and output for model"""
def prepare_sample_multipleinputs(features):
    #image = tf.image.resize(features["image"], size=(224, 224))
    #return image, features["category_id"]
    left_ax = features["left_ax"]
    left_ay = features["left_ay"]
    left_az = features["left_az"]
    #left_gx = features["left_gx"]
    #left_gy = features["left_gy"]
    #left_gz = features["left_gz"]
    output_data = features['score_10k']
    
    return (left_ax, left_ay, left_az), output_data

"""
(tf.data.TFRecordDataset(filenames)
 .map(parse_tfrecord_fn)
 .map(prepare_sample_multipleinputs)
)
"""

"""fetch the data"""
AUTOTUNE = tf.data.AUTOTUNE

"""
for batch in tf.data.TFRecordDataset(train_filenames).map(parse_tfrecord_fn):
    tens = batch
    print(batch)
    break
"""

def get_dataset_multipleinputs(filenames, batch_size):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample_multipleinputs, num_parallel_calls=AUTOTUNE)
        .shuffle(batch_size * 10)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset


get_dataset_multipleinputs(train_filenames, batch_size)


"""create neural network """
input_1 = layers.Input(shape=(5000, 1, 1), name = 'input_1')
input_2 = layers.Input(shape=(5000, 1, 1), name = 'input_2')
input_3 = layers.Input(shape=(5000, 1, 1), name = 'input_3')

reshape_input_1 = layers.Reshape((1,5000,1), name = 'reshape_input_1')(input_1)
reshape_input_2 = layers.Reshape((1,5000,1), name = 'reshape_input_2')(input_2)
reshape_input_3 = layers.Reshape((1,5000,1), name = 'reshape_input_3')(input_3)

conv1d_1 = layers.Conv1D(16, 1001, activation = 'relu', name = 'conv1d_1')(reshape_input_1) 
# output: 4000x1x16
reshape_1 = layers.Reshape((4000, 16, 1), name = 'reshape_1')(conv1d_1)
conv2d_1 = layers.Conv2D(32, (1001,3), activation = 'relu', name = 'conv2d_1')(reshape_1) 
# output: 3000x14x32
maxpool_1 = layers.MaxPool2D(2, name = 'maxpool_1')(conv2d_1) 
flatten_1 = layers.Flatten(name = 'flatten_1')(maxpool_1)


conv1d_2 = layers.Conv1D(16, 1001, activation = 'relu', name = 'conv1d_2')(reshape_input_2) 
# output: 4000x1x16
reshape_2 = layers.Reshape((4000, 16, 1), name = 'reshape_2')(conv1d_2)
conv2d_2 = layers.Conv2D(32, (1001,3), activation = 'relu', name = 'conv2d_2')(reshape_2) 
# output: 3000x14x32
maxpool_2 = layers.MaxPool2D(2, name = 'maxpool_2')(conv2d_2)
flatten_2 = layers.Flatten(name = 'flatten_2')(maxpool_2)


conv1d_3 = layers.Conv1D(16, 1001, activation = 'relu', name = 'conv1d_3')(reshape_input_3) 
# output: 4000x1x16
reshape_3 = layers.Reshape((4000, 16, 1), name = 'reshape_3')(conv1d_3)
conv2d_3 = layers.Conv2D(32, (1001,3), activation = 'relu', name = 'conv2d_3')(reshape_3) 
# output: 3000x14x32
maxpool_3 = layers.MaxPool2D(2, name = 'maxpool_3')(conv2d_3)
flatten_3 = layers.Flatten(name = 'flatten_3')(maxpool_3)

concat_1_2_3 = layers.Concatenate(name = 'concat_1_2_3')([flatten_1, flatten_2, flatten_3])

dense1 = layers.Dense(16, activation = 'relu', name = 'dense1')(concat_1_2_3)
output = layers.Dense(1, activation = 'relu', name = 'output')(dense1)

model_mi = models.Model(inputs=[input_1, input_2, input_3], outputs=[output])

model_mi.summary()

plot_model(model_mi, show_shapes=True, show_layer_names=True)

model_mi.compile(loss='mean_absolute_error',
       optimizer='adam', metrics=['mean_absolute_error'])

examples_per_file = 27

steps_per_epoch = np.int(np.ceil(examples_per_file*len(train_filenames)/batch_size))
validation_steps = np.int(np.ceil(examples_per_file*len(val_filenames)/batch_size))
steps = np.int(np.ceil(examples_per_file*len(test_filenames)/batch_size))
print("steps_per_epoch = ", steps_per_epoch)
print("validation_steps = ", validation_steps)
print("steps = ", steps)

train_dataset_mi = get_dataset_multipleinputs(train_filenames, batch_size)
val_dataset_mi = get_dataset_multipleinputs(val_filenames, batch_size)
test_dataset_mi = get_dataset_multipleinputs(test_filenames, batch_size)

train_dataset_mi

steps_per_epoch = steps_per_epoch

model_mi.fit(train_dataset_mi,
          validation_data = val_dataset_mi, 
          steps_per_epoch = steps_per_epoch,
          validation_steps = validation_steps, 
          epochs = epochs
         )

model_mi.evaluate(test_dataset_mi, steps = len(test_filenames))

steps_to_take = len(test_filenames) #1100



#get_dataset(filenames, batch_size)



"""calculate accuracy in different ways"""
evaluated_accuracy = model_mi.evaluate(test_dataset_mi)
evaluated_accuracy
pred_values = model_mi.predict(test_dataset_mi)

steps_to_take = len(test_filenames)

pred_values_list = []
pred_list = []
true_list = []
"""
for x, y in test_dataset.take(steps_to_take):
    
    pred_value = model.predict(x)
    pred_values_list = pred_values_list + list(pred_value)
"""


real_data = list(test_dataset_mi.as_numpy_iterator())

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
    

    
