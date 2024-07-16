#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:31:23 2022

@author: patrickmayerhofer

modeling
"""

import tensorflow as tf
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
import keras
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve, PrecisionRecallDisplay


"""import and shuffle"""
tfrecords_dir = "/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Running Plantiga Project/Code/time_series_tfrecords_example/working/tfrecords"

filenames = tf.io.gfile.glob(f"{tfrecords_dir}/*.tfrec")

shuffled_filenames = filenames.copy()
np.random.shuffle(shuffled_filenames)

print(f"Total: {len(shuffled_filenames)}")
print("---------")


"""create training and validation sizes and filenames"""
train_val_size = 0.2
train_size = 0.2

train_val_len = int(len(shuffled_filenames)*train_val_size)
train_val_filenames, test_filenames = shuffled_filenames[:train_val_len], shuffled_filenames[train_val_len:]

train_len = int(len(train_val_filenames)*train_size)
train_filenames, val_filenames = train_val_filenames[:train_len], train_val_filenames[train_len:]

print(f"Train: {len(train_filenames)}")
print(f"Validation: {len(val_filenames)}")
print(f"Test: {len(test_filenames)}")


for batch in tf.data.TFRecordDataset(filenames):
    print(batch)
    break


"""parse serialized data"""
def parse_tfrecord_fn(example):
    
    feature_description = {
        "acc_x": tf.io.VarLenFeature(tf.float32),
        "acc_y": tf.io.VarLenFeature(tf.float32),
        "acc_z": tf.io.VarLenFeature(tf.float32),
        "acc_matrix": tf.io.VarLenFeature(tf.float32),
        "id": tf.io.FixedLenFeature([], tf.int64),
        "fullpath": tf.io.FixedLenFeature([], tf.string),
        "target_id": tf.io.FixedLenFeature([], tf.int64),
        'target_id_onehot': tf.io.VarLenFeature(tf.float32)
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["acc_x"] = tf.sparse.to_dense(example["acc_x"])
    example["acc_y"] = tf.sparse.to_dense(example["acc_y"])
    example["acc_z"] = tf.sparse.to_dense(example["acc_z"])
    example["acc_matrix"] = tf.reshape(tf.sparse.to_dense(example["acc_matrix"]), (30,3,1))
    example["target_id_onehot"] = tf.sparse.to_dense(example["target_id_onehot"])
    return example

for batch in tf.data.TFRecordDataset(filenames).map(parse_tfrecord_fn):
    print(batch)
    break


"""prepare input and output for model"""
def prepare_sample(features):
    #image = tf.image.resize(features["image"], size=(224, 224))
    #return image, features["category_id"]
    acc_all = features["acc_matrix"]
    target = features['target_id_onehot']
    
    return acc_all, target

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


#get_dataset(filenames, batch_size)

"""fun part"""
multiclass_problem = True

final_activation = 'sigmoid'
final_layer_size = 1
if multiclass_problem:
    final_activation = 'softmax'
    final_layer_size = 4

model = tf.keras.Sequential([
    layers.Conv2D(16, 3, activation = "relu", input_shape = (30,3,1)),
    #layers.MaxPool2D(2),
    layers.Reshape((28,16,1)),
    layers.Conv2D(32, 3, activation = "relu"),
    layers.MaxPool2D(2),
    layers.Flatten(),
    layers.Dense(16, activation = "relu"),
    layers.Dense(final_layer_size, activation = final_activation) # softmax va bene per i multiclass, altrimenti uso sigmoid
])
model.summary()

loss_to_use = tf.keras.losses.BinaryCrossentropy()
if multiclass_problem:
    loss_to_use = 'categorical_crossentropy'

model.compile(loss = loss_to_use, 
              optimizer = "adam", 
              metrics = ["accuracy", 
                         tf.keras.metrics.AUC(curve = 'ROC'),
                         tf.keras.metrics.AUC(curve = 'PR'),
                         tf.keras.metrics.Precision(),
                         tf.keras.metrics.Recall(),
                         tf.keras.metrics.PrecisionAtRecall(0.8) 
                        ]) #tf.keras.metrics.AUC(from_logits=True)


get_dataset(filenames, batch_size)


examples_per_file = 128

steps_per_epoch = np.int(np.ceil(examples_per_file*len(train_filenames)/batch_size))
validation_steps = np.int(np.ceil(examples_per_file*len(val_filenames)/batch_size))
steps = np.int(np.ceil(examples_per_file*len(test_filenames)/batch_size))
print("steps_per_epoch = ", steps_per_epoch)
print("validation_steps = ", validation_steps)
print("steps = ", steps)

train_dataset = get_dataset(train_filenames, batch_size)
val_dataset = get_dataset(val_filenames, batch_size)
test_dataset = get_dataset(test_filenames, batch_size)

epochs = 50
steps_per_epoch = steps_per_epoch

model.fit(train_dataset,
          validation_data = val_dataset, 
          steps_per_epoch = steps_per_epoch,
          validation_steps = validation_steps, 
          epochs = epochs
         )

acc = model.evaluate(test_dataset, steps = len(test_filenames))
pred = model.predict(test_dataset)

steps_to_take = len(test_filenames)

pred_values_list = []
pred_list = []
true_list = []
true_list_onehot = []

for x, y in test_dataset.take(steps_to_take):
    
    pred_value = model.predict(x)
    if multiclass_problem:
        pred = pred_value.argmax(1)
    else:
        threshold = 0.5
        pred = pred_value > threshold
    
    pred_values_list = pred_values_list + list(pred_value)
    pred_list = pred_list + list(pred)
    true_list = true_list + list(y.numpy().argmax(axis=1).astype(int))
    true_list_onehot = true_list_onehot + list(y.numpy().astype(int))
    
print('Accuracy')
print(accuracy_score(true_list, [x.astype(int) for x in pred_list]))

print('Confusion Matrix')
print(confusion_matrix(true_list, [x.astype(int) for x in pred_list]))

m = tf.keras.metrics.AUC(curve = 'PR')
m.update_state(true_list_onehot, pred_values_list)
m.result().numpy()