#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:48:58 2023

@author: patrick
"""

import tensorflow as tf
import numpy as np
from keras import layers, models
from keras.layers import LSTM, Dense,Flatten,Conv3D
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import keras
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt


data_name = "8k_250_0"
subjects = [1,2,3,4,5,6,7,8,9,10]
val_split = 0.2
test_split = 0.2
flag_shuffle_train = 0
flag_plot = 0
epochs = 5
dropout = 0.2
weights_to_use = "imagenet"
input_my_model = keras.Input(shape = (126,32,12,1)) #keras.Input(shape = (126,40,12,1))  #keras.Input(shape = (10000,12)) # input for first layer
input_resnet = keras.Input(shape=(126,32,3)) # input after my own layers into the resnet
resnet_trainable = True
final_layer_size = 4



"""import and shuffle"""
dir_root = "/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Running Plantiga Project/Data/"
dir_tfrecords = dir_root + 'Prepared/tfrecords/' + data_name + '/Treadmill/'


test_subjects = subjects[0:int(len(subjects)*test_split)]
val_subjects = subjects[int(len(subjects)*test_split):int(len(subjects)*test_split) + int(len(subjects)*val_split)]
train_subjects = subjects[int(len(subjects)*test_split) + int(len(subjects)*val_split):len(subjects)]

train_filenames = list()
val_filenames = list()
test_filenames = list()
speeds = [0,1,2]
for subject in subjects:
    for speed in speeds:
        sensor = "SENSOR" + "{:03d}".format(subject)
        dir_tfr_data = dir_tfrecords + 'speed' + str(speed) + '/' + sensor + ".tfrecords"
        if subject in test_subjects:
            test_filenames.append(dir_tfr_data)
        elif subject in val_subjects:
            val_filenames.append(dir_tfr_data)   
        else:
            train_filenames.append(dir_tfr_data)
if flag_shuffle_train:
    np.random.shuffle(train_filenames)



print(f"Train: {len(train_filenames)}")
print(f"Validation: {len(val_filenames)}")
print(f"Test: {len(test_filenames)}")


for batch in tf.data.TFRecordDataset(train_filenames):
    print(batch)
    break


def parse_tfrecord_image(example):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  feature_description = {
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

  example = tf.io.parse_single_example(example, feature_description)  
  
  example['height'] = example['height']
  example['width'] = example['width']
  example['depth'] = example['depth']
  example['height'] = example['height']
  example['score_10k'] = example['score_10k']
  example['seconds_10k'] = example['seconds_10k']
  
  example['spectrogram_image'] = tf.io.parse_tensor(example['spectrogram_image'], out_type=tf.double)
  example['spectrogram_image'] = tf.reshape(example['spectrogram_image'], shape=[example['height'],example['width'],example['depth']])
  
  example['bin_label'] = example['bin_label']
  example['bin_label_onehot'] = tf.one_hot(example['bin_label'], depth = 2, dtype = 'int64')
  
  example['subject_id'] = example['subject_id']
  example['subject_id_onehot'] = tf.one_hot(example['subject_id']-1, depth = 188, dtype = 'int64')

  
  example['speed_label'] = example['speed_label']
  example['speed_label_onehot'] = tf.one_hot(example['speed_label'], depth = 4, dtype = 'int64')
 
  
  return example


tense = list()
for batch in tf.data.TFRecordDataset(train_filenames).map(parse_tfrecord_image):
    #print(batch)
    tense.append(batch)
    #break



"""prepare input and output for model"""
def prepare_sample(features):
    #image = tf.image.resize(features["image"], size=(224, 224))
    #return image, features["category_id"]
    spectrogram_image = features["spectrogram_image"]
    speed_onehot = features['speed_label_onehot']
    
    return spectrogram_image, speed_onehot




(tf.data.TFRecordDataset(train_filenames)
 .map(parse_tfrecord_image)
 .map(prepare_sample)
)


# my own stuff: plot some data
if flag_plot:
    print('not implemented yet')
             
  

"""fetch the data"""
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32


def get_dataset(filenames, batch_size):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_image, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=AUTOTUNE)
        .shuffle(batch_size * 10)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset




get_dataset(train_filenames, batch_size)

"""fun part"""
final_activation = 'sigmoid'
if final_layer_size > 1:
    final_activation = 'softmax'

if 1:
    # resnet50 with weights from "imagenet" dataset
    res_model_pretrained = keras.applications.ResNet50(include_top = False, #so that we can change input and output layer
                                            weights=weights_to_use, 
                                            input_tensor=input_resnet)
    model = Sequential()
    model.add(input_my_model)
    model.add(Conv3D(filters=16, kernel_size=(1,1,3), strides=(1,1,1), activation= 'relu'))
    model.add(Conv3D(filters=32, kernel_size=(1,1,3), strides=(1,1,1), activation= 'relu'))
    model.add(Conv3D(filters=64, kernel_size=(1,1,3), strides=(1,1,1), activation= 'relu'))
    model.add(Conv3D(filters=3, kernel_size=(1,1,3), strides=(1,1,1), activation= 'relu'))
    model.add(Conv3D(filters=1, kernel_size=(1,1,2), strides=(1,1,1), activation= 'relu'))
    model.add(res_model_pretrained)
    model.add(Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(final_layer_size, activation=final_activation))
    model.summary()
    
 
    
 
loss_to_use = tf.keras.losses.BinaryCrossentropy()
if final_layer_size > 1:
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

loss_to_use = tf.keras.losses.BinaryCrossentropy()
if final_layer_size > 1:
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


model.layers[5].trainable = resnet_trainable

# check which parts overall are frozen
for i, layer in enumerate(model.layers):
    print(i, layer.name, "-", layer.trainable)

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
test_dataset = get_dataset(test_filenames, batch_size)


#steps_per_epoch = steps_per_epoch

model.fit(train_dataset,
          validation_data = val_dataset, 
          #steps_per_epoch = steps_per_epoch,
          #validation_steps = validation_steps, 
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
    if final_layer_size > 1:
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