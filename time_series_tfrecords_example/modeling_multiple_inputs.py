#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:31:23 2022

@author: patrickmayerhofer

modeling
"""

import tensorflow as tf
from tensorflow.keras import layers, models 
import numpy as np
from keras.utils.vis_utils import plot_model
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
train_val_size = 0.7
train_size = 0.7

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
def prepare_sample_multipleinputs(features):
    #image = tf.image.resize(features["image"], size=(224, 224))
    #return image, features["category_id"]
    acc_x = features["acc_x"]
    acc_y = features["acc_y"]
    target = features['target_id_onehot']
    
    return (acc_x, acc_y), target


(tf.data.TFRecordDataset(filenames)
 .map(parse_tfrecord_fn)
 .map(prepare_sample_multipleinputs)
)

"""fetch the data"""
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32


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

"""fun part"""
multiclass_problem = True

final_activation = 'sigmoid'
final_layer_size = 1
if multiclass_problem:
    final_activation = 'softmax'

### Can't use Sequential API with multiple inputs
    
# model = tf.keras.Sequential([
#     layers.Conv2D(16, 1, activation = "relu", input_shape = (30,1,1)),
#     #layers.MaxPool2D(2),
#     layers.Reshape((30,16,1)),
#     layers.Conv2D(32, 3, activation = "relu"),
#     layers.MaxPool2D(2),
#     layers.Flatten(),
#     layers.Dense(16, activation = "relu"),
#     layers.Dense(final_layer_size, activation = final_activation) # softmax va bene per i multiclass, altrimenti uso sigmoid
# ])

input_x = layers.Input(shape=(30, 1, 1), name = 'input_x')
input_y = layers.Input(shape=(30, 1, 1), name = 'input_y')

reshape_input_x = layers.Reshape((1,30,1), name = 'reshape_input_x')(input_x)
reshape_input_y = layers.Reshape((1,30,1), name = 'reshape_input_y')(input_y)

conv1d_x = layers.Conv1D(16, 3, activation = 'relu', name = 'conv1d_x')(reshape_input_x) # output: 28x1x16
reshape_x = layers.Reshape((28, 16, 1), name = 'reshape_x')(conv1d_x)
conv2d_x = layers.Conv2D(32, 3, activation = 'relu', name = 'conv2d_x')(reshape_x) # output: 26x14x32
maxpool_x = layers.MaxPool2D(2, name = 'maxpool_x')(conv2d_x) # output:
flatten_x = layers.Flatten(name = 'flatten_x')(maxpool_x)

conv1d_y = layers.Conv1D(16, 3, activation = 'relu', name = 'conv1d_y')(reshape_input_y) # output: 28x1x16
reshape_y = layers.Reshape((28, 16, 1), name = 'reshape_y')(conv1d_y)
conv2d_y = layers.Conv2D(32, 3, activation = 'relu', name = 'conv2d_y')(reshape_y) # output: 26x14x32
maxpool_y = layers.MaxPool2D(2, name = 'maxpool_y')(conv2d_y)
flatten_y = layers.Flatten(name = 'flatten_y')(maxpool_y)

concat_x_y = layers.Concatenate(name = 'concat_x_y')([flatten_x, flatten_y])

dense1 = layers.Dense(16, activation = 'relu', name = 'dense1')(concat_x_y)
output = layers.Dense(4, activation = final_activation, name = 'output')(dense1)

model_mi = models.Model(inputs=[input_x, input_y], outputs=[output])

model_mi.summary()

plot_model(model_mi, show_shapes=True, show_layer_names=True)

loss_to_use = tf.keras.losses.BinaryCrossentropy()
if multiclass_problem:
    loss_to_use = 'categorical_crossentropy'

model_mi.compile(loss = loss_to_use, 
              optimizer = "adam", 
              metrics = ["accuracy", 
                         tf.keras.metrics.AUC(curve = 'ROC'),
                         tf.keras.metrics.AUC(curve = 'PR'),
                         tf.keras.metrics.Precision(),
                         tf.keras.metrics.Recall(),
                         tf.keras.metrics.PrecisionAtRecall(0.8) 
                        ]) #tf.keras.metrics.AUC(from_logits=True)


#get_dataset(filenames, batch_size)


examples_per_file = 128

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

epochs = 10
steps_per_epoch = steps_per_epoch

model_mi.fit(train_dataset_mi,
          validation_data = val_dataset_mi, 
          steps_per_epoch = steps_per_epoch,
          validation_steps = validation_steps, 
          epochs = epochs
         )

model_mi.evaluate(test_dataset_mi, steps = len(test_filenames))

steps_to_take = len(test_filenames) #1100

pred_values_list = []
pred_list = []
true_list = []
true_list_onehot = []

for x, y in test_dataset_mi.take(steps_to_take):
    
    pred_value = model_mi.predict(x)
    if multiclass_problem:
        pred = pred_value.argmax(1) ## solo per multiclasse
    else:
        threshold = 0.5
        pred = pred_value > threshold
    # TODO aggiungere una if per determinare se il problema è multiclasse o binario
    #print(pred)
    #print(y)
    
    pred_values_list = pred_values_list + list(pred_value)
    pred_list = pred_list + list(pred)
    true_list = true_list + list(y.numpy().argmax(axis=1).astype(int))
    true_list_onehot = true_list_onehot + list(y.numpy().astype(int))
    
#print(pred_list)
#print(true_list)

print('Accuracy')
print(accuracy_score(true_list, [x.astype(int) for x in pred_list]))

print('Confusion Matrix')
print(confusion_matrix(true_list, [x.astype(int) for x in pred_list]))

