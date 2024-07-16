#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:48:58 2023

@author: patrick
"""

import tensorflow as tf
import numpy as np
from keras import layers, models
from keras.models import Sequential
from keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
import keras
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt


data_name = "5k_no_spectrogram"
subjects = [1,2,3,8,9,20,21,24,28,32,33,35,36,42,45,49,55,79,108,110,111,122,131,133]
flag_shuffle_train = 0
flag_plot = 0
epochs = 5
trials = [1,2,3]


"""import and shuffle"""
dir_root = "/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Running Plantiga Project/Data/"
dir_tfrecords = dir_root + 'Prepared/tfrecords/' + data_name + '/'

train_filenames = list()
val_filenames = list()


speeds = [0,1,2]
for subject in subjects:
    sensor = "SENSOR" + "{:03d}".format(subject)
    for trial in trials:
        if trial == 1:
            for speed in speeds:
                dir_tfr_data = dir_tfrecords + 'Treadmill/' + 'speed' + str(speed) + '/' + sensor + ".tfrecords"
                train_filenames.append(dir_tfr_data)
            dir_tfr_data = dir_tfrecords + 'Overground/' + sensor + ".tfrecords"
            val_filenames.append(dir_tfr_data)
        else:
            for speed in speeds:
                dir_tfr_data = dir_tfrecords + 'Treadmill/' + 'speed' + str(speed) + '/' + sensor + '_' + str(trial) + ".tfrecords"
                train_filenames.append(dir_tfr_data)
            dir_tfr_data = dir_tfrecords + 'Overground/' + sensor + '_' + str(trial) + ".tfrecords"
            val_filenames.append(dir_tfr_data)
        
if flag_shuffle_train:
    np.random.shuffle(train_filenames)



print(f"Train: {len(train_filenames)}")
print(f"Validation: {len(val_filenames)}")
#print(f"Test: {len(test_filenames)}")


for batch in tf.data.TFRecordDataset(train_filenames):
    print(batch)
    break


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
        "speed": tf.io.FixedLenFeature([], tf.int64),
        'speed_onehot': tf.io.VarLenFeature(tf.float32),
        'subject_id_onehot': tf.io.VarLenFeature(tf.float32)
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
    #example['subject_id'] = tf.one_hot(example['subject_id']-1, depth = 2, dtype = 'int64')
    example["subject_id"] = example['subject_id']
    example["speed"] = example["speed"]
    example["speed_onehot"] = tf.sparse.to_dense(example["speed_onehot"])
    example["subject_id_onehot"] = tf.sparse.to_dense(example["subject_id_onehot"])
    
    
    #example["speed"] = tf.one_hot(example['speed']-1, depth = 3, dtype = 'int64')
    return example

tense = list()
for batch in tf.data.TFRecordDataset(train_filenames).map(parse_tfrecord_rnn):
    #print(batch)
    tense.append(batch)
    #break



"""prepare input and output for model"""
def prepare_sample(features):
    #image = tf.image.resize(features["image"], size=(224, 224))
    #return image, features["category_id"]
    acc_left_ay = features["left_ay"]
    target = features['subject_id_onehot']
    
    return acc_left_ay, target

def prepare_sample_multipleinputs(features):
    #image = tf.image.resize(features["image"], size=(224, 224))
    #return image, features["category_id"]
    left_ay = features["left_ay"]
    left_az = features["left_az"]
    target = features['subject_id_onehot']
    
    return (left_ay, left_az), target

(tf.data.TFRecordDataset(train_filenames)
 .map(parse_tfrecord_rnn)
 .map(prepare_sample)
)

# my own stuff: plot some data
if flag_plot:
    figure_id_new = 0 
    figure_id_previous = 0
    s_id_previous = 0
    sp_id_previous = 0
    for i in range(0,len(tense)):
        s_id = int(tense[i]['subject_id'])
        sp_id = int(tense[i]['speed'])
        if i == 0:
            plt.figure(figure_id_new)
            
            
        if s_id == s_id_previous and sp_id == sp_id_previous:
            plt.figure(figure_id_previous)
            
        if s_id != s_id_previous or sp_id != sp_id_previous:
            figure_id_new = figure_id_new + 1
            plt.figure(figure_id_new)
            
        plt.title('Subject: ' + str(s_id) + ', Speed: ' + str(sp_id))
        plt.plot(tense[i]['left_ay'])
        
        figure_id_previous = figure_id_new
        s_id_previous = s_id
        sp_id_previous = sp_id
             
  

"""fetch the data"""
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32


def get_dataset(filenames, batch_size):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_rnn, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=AUTOTUNE)
        .shuffle(batch_size * 10)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset


get_dataset(train_filenames, batch_size)

"""fun part"""
multiclass_problem = True

final_activation = 'sigmoid'
final_layer_size = 1
if multiclass_problem:
    final_activation = 'softmax'
    final_layer_size = 10
if 1:
    model = tf.keras.Sequential([
        layers.Conv1D(16, 500, activation = "relu", input_shape = (5000,1)),
        #layers.MaxPool2D(2),
        layers.Reshape((4501,16,1)),
        layers.Conv2D(32, [500,3], activation = "relu"),
        layers.MaxPool2D(2),
        layers.Flatten(),
        layers.Dense(16, activation = "relu"),
        layers.Dense(final_layer_size, activation = final_activation) # softmax va bene per i multiclass, altrimenti uso sigmoid
    ])
    model.summary()
    
if 0:
    model = Sequential()
    model.add(LSTM(64, activation='relu', return_sequences=False, input_shape = (5000,1)))
    model.add(layers.Dense(final_layer_size, activation = final_activation))
    
    print(model.summary())

if 0:
    input_y = layers.Input(shape=(5000, 1), name = 'input_y')
    input_z = layers.Input(shape=(5000, 1), name = 'input_z')

    conv1d_y = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_y')(input_y) # output: 28x1x16
    reshape_y = layers.Reshape((4501, 16, 1), name = 'reshape_y')(conv1d_y)
    conv2d_y = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_y')(reshape_y) # output: 26x14x32
    maxpool_y = layers.MaxPool2D(5, name = 'maxpool_y')(conv2d_y)
    flatten_y = layers.Flatten(name = 'flatten_y')(maxpool_y)
    
    conv1d_z = layers.Conv1D(16, 500, activation = 'relu', name = 'conv1d_z')(input_z) # output: 28x1x16
    reshape_z = layers.Reshape((4501, 16, 1), name = 'reshape_z')(conv1d_z)
    conv2d_z = layers.Conv2D(32, [500,3], activation = 'relu', name = 'conv2d_z')(reshape_z) # output: 26x14x32
    maxpool_z = layers.MaxPool2D(5, name = 'maxpool_z')(conv2d_z) # output:
    flatten_z = layers.Flatten(name = 'flatten_z')(maxpool_z)

    

    concat_y_z = layers.Concatenate(name = 'concat_y_z')([flatten_y, flatten_z])

    dense1 = layers.Dense(64, activation = 'relu', name = 'dense1')(concat_y_z)
    dense2 = layers.Dense(32, activation = 'relu', name = 'dense2')(dense1)
    dense3 = layers.Dense(16, activation = 'relu', name = 'dense3')(dense2)
    output = layers.Dense(3, activation = final_activation, name = 'output')(dense3)

    model = models.Model(inputs=[input_y, input_z], outputs=[output])

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


#get_dataset(filenames, batch_size)


#examples_per_file = 128

#steps_per_epoch = int(np.ceil(examples_per_file*len(train_filenames)/batch_size))
#validation_steps = int(np.ceil(examples_per_file*len(val_filenames)/batch_size))
#steps = int(np.ceil(examples_per_file*len(test_filenames)/batch_size))
#print("steps_per_epoch = ", steps_per_epoch)
#print("validation_steps = ", validation_steps)
#print("steps = ", steps)

train_dataset = get_dataset(train_filenames, batch_size)
val_dataset = get_dataset(val_filenames, batch_size)


#steps_per_epoch = steps_per_epoch

model.fit(train_dataset,
          validation_data = val_dataset, 
          #steps_per_epoch = steps_per_epoch,
          #validation_steps = validation_steps, 
          epochs = epochs
         )

acc = model.evaluate(val_dataset, steps = len(val_filenames))
pred = model.predict(val_dataset)

steps_to_take = len(val_filenames)

pred_values_list = []
pred_list = []
true_list = []
true_list_onehot = []

for x, y in val_dataset.take(steps_to_take):
    
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
    
print('Accuracy_val_dataset')
print(accuracy_score(true_list, [x.astype(int) for x in pred_list]))

print('Confusion Matrix Val Dataset')
print(confusion_matrix(true_list, [x.astype(int) for x in pred_list]))

m = tf.keras.metrics.AUC(curve = 'PR')
m.update_state(true_list_onehot, pred_values_list)
m.result().numpy()