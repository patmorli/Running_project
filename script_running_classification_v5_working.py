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


"""some variables"""
n_timesteps = 30 # predefined, when we saved tfrecords in "create_TFRecords3"
n_features = 12 # same here. 3 accelerations, 3 angular velocities, 2 feet
examples_per_file = 128 # same here, these are the nr of files in one tfrecord
epochs = 50

"""import and shuffle"""
dir_root = '/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Data/Prepared/'
dir_tfr = dir_root + "tfrecords/"

filenames = tf.io.gfile.glob(f"{dir_tfr}*.tfrec")

shuffled_filenames = filenames.copy()
np.random.shuffle(shuffled_filenames)

print(f"Total: {len(shuffled_filenames)}")
print("---------")


"""create training and validation sizes and filenames"""
train_val_size = 0.8 # 80% for training, 20% for testing
train_size = 0.8 # of those 80%, 80% will be for training, 20% will be for validation

train_val_len = int(len(shuffled_filenames)*train_val_size)
train_val_filenames, test_filenames = shuffled_filenames[:train_val_len], shuffled_filenames[train_val_len:]

train_len = int(len(train_val_filenames)*train_size)
train_filenames, val_filenames = train_val_filenames[:train_len], train_val_filenames[train_len:]

print(f"Train: {len(train_filenames)}")
print(f"Validation: {len(val_filenames)}")
print(f"Test: {len(test_filenames)}")

"""
for batch in tf.data.TFRecordDataset(filenames):
    print(batch)
    break
"""

"""parse serialized data"""
def parse_tfrecord_fn(example):
    
    feature_description = {
        "feature_matrix": tf.io.VarLenFeature(tf.float32),
        "score_10k": tf.io.FixedLenFeature([], tf.int64),
        "fullpath": tf.io.FixedLenFeature([], tf.string),
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

"""fetch the data"""
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32

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


get_dataset(filenames, batch_size)


"""create neural network """
model = tf.keras.Sequential([
    layers.Conv2D(16, 12, activation = "relu", input_shape = (30,12,1)),
    #layers.MaxPool2D(2),
    layers.Reshape((19,16,1)),
    layers.Conv2D(32, 3, activation = "relu"),
    layers.MaxPool2D(2),
    layers.Flatten(),
    layers.Dense(16, activation = "relu"),
    layers.Dense(1, activation = "relu") # softmax va bene per i multiclass, altrimenti uso sigmoid
])
model.summary()

model.compile(loss='MeanSquaredError',
       optimizer='adam', metrics=['MeanSquaredError'])


get_dataset(filenames, batch_size)


"""
steps_per_epoch = np.int(np.ceil(examples_per_file*len(train_filenames)/batch_size))
validation_steps = np.int(np.ceil(examples_per_file*len(val_filenames)/batch_size))
steps = np.int(np.ceil(examples_per_file*len(test_filenames)/batch_size))
print("steps_per_epoch = ", steps_per_epoch)
print("validation_steps = ", validation_steps)
print("steps = ", steps)
"""

train_dataset = get_dataset(train_filenames, batch_size)
val_dataset = get_dataset(val_filenames, batch_size)
test_dataset = get_dataset(test_filenames, batch_size)


model.fit(train_dataset,
          validation_data = val_dataset, 
          #steps_per_epoch = steps_per_epoch,
          #validation_steps = validation_steps, 
          epochs = epochs
         )


model.evaluate(test_dataset, steps = len(test_filenames))
pred_values = model.predict(test_dataset)

steps_to_take = len(test_filenames)

pred_values_list = []
pred_list = []
true_list = []
true_list_onehot = []

for x, y in test_dataset.take(steps_to_take):
    
    pred_value = model.predict(x).astype(int)
    
    pred_values_list = pred_values_list + list(pred_value)
    true_list = true_list + list(y.numpy().astype(int))



    
print('Accuracy')
#print(accuracy_score(true_list, [x.astype(int) for x in pred_list]))

print('Confusion Matrix')
#print(confusion_matrix(true_list, [x.astype(int) for x in pred_list]))

m = tf.keras.metrics.AUC(curve = 'PR')
m.result().numpy()