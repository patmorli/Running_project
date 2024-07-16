#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 12:21:55 2022

@author: patrickmayerhofer

functions_running_classification
"""
from scipy import stats
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Example, Features, Feature




"""Splits the full dataset into windows.
Input: X, y, length of window, sliding length"""
def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X)-step, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

"""Splits the full dataset into windows. 
Different from the previous as in y and x do not matter
Input: data, length of window, sliding length"""
def create_windows(data, time_steps=1, step=1):
    data_s = []
    for i in range(0, len(data)-step, step):
        v = data.iloc[i:(i + time_steps)].values
        data_s.append(v)
    return data_s


"""create a custom model for classification.
train X is the input data. layers_nodes is a list where the 
size represents the number of layers and the numbers of nodes in each layer"""
def custom_model(trainX, layers_nodes, learning_rate):
    # model_specifics is a list where the size represents the number of
    # layers and the numbers the nodes in each layer
    model = Sequential()
    model.add(LSTM(layers_nodes[0], activation='relu', return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
    for i in range(0,len(layers_nodes)-1):
        model.add(LSTM(layers_nodes[i+1], activation='relu', return_sequences=True))
        
    model.add(layers.Dense(1, activation = 'linear'))
    
    model.compile(optimizer = Adam(learning_rate = learning_rate), loss='mse', metrics = ['mse'])
    print(model.summary())
    return model

"""Create the bidirectional lstm model. Needs X_train and 
y_train to understand the nature of the model input and output.
Input: X_train, ytrain 
Output: The prepared model"""
def create_model(X_train, y_train):
    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional(
          keras.layers.LSTM(
              units=128,
              input_shape=[X_train.shape[1], X_train.shape[2]]
          )
        )
    )
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
  
    return model


"""Plot a confusion matrix
Input: y_true, y_predicted, class_names"""     
def plot_cm(y_true, y_pred, class_names):
      cm = confusion_matrix(y_true, y_pred)
      fig, ax = plt.subplots(figsize=(18, 16)) 
      ax = sns.heatmap(
          cm, 
          annot=True, 
          fmt="d", 
          cmap=sns.diverging_palette(220, 20, n=7),
          ax=ax
      )
    
      plt.ylabel('Actual')
      plt.xlabel('Predicted')
      ax.set_xticklabels(class_names)
      ax.set_yticklabels(class_names)
      b, t = plt.ylim() # discover the values for bottom and top
      b += 0.5 # Add 0.5 to the bottom
      t -= 0.5 # Subtract 0.5 from the top
      plt.ylim(b, t) # update the ylim(bottom, top) values
      plt.show() # ta-da!
      
      
"""Treadmill import data"""
def import_cut_treadmill(subjects, trials, speeds, dir_data_treadmill):
    # creates lists of lists of lists (3 layers)
    # data_treadmill_key holds the names of each dataset from data_treadmill_cut
    # first layer: subjects, second layer: trials, third layer: speeds, fourth layer: actual data
    
    data_treadmill_key = list('0') # index place 0 with 0 to store stubject 1 at index 1
    data_treadmill_cut = list('0')
    
    for subject_id in subjects:
        subject_name = 'SENSOR' + str(subject_id).zfill(3)
        data_cut_trials = list()
        data_key_trials = list()
        for trial_id in trials:
            data_cut_speeds = list()
            data_key_speeds = list()
            for speed in speeds:
                if trial_id == 1:
                    key = subject_name + '_' + str(speed)
                else:
                    key = subject_name + '_' + str(speed) + '_' + str(trial_id)
        
                # import dataset
                data = pd.read_csv(dir_data_treadmill + key + '.csv')
                # delete unusable data (usabledata = False)
                data = data[data['usabledata'] == True]
                
                data_cut_speeds.append(data)
                data_key_speeds.append(key) 
                
            data_key_trials.append(data_key_speeds)
            data_cut_trials.append(data_cut_speeds)
    
        data_treadmill_key.append(data_key_trials)
        data_treadmill_cut.append(data_cut_trials) 
        
    return data_treadmill_cut, data_treadmill_key


"""overground import data"""
def import_cut_overground(subjects, trials, dir_data_overground):
    # creates lists of lists of lists (3 layers)
    # data_overground_raw_key holds the names of each dataset from data_overground_raw
    # first layer: subjects, second layer: trials, third layer: both directions
    data_overground_key = list('0') # index place 0 with 0 to store stubject 1 at index 1
    data_overground_cut = list('0')
    for subject_id in subjects:
        subject_name = 'SENSOR' + str(subject_id).zfill(3)
        data_cut_trials = list()
        data_key_trials = list()
        
        for trial_id in trials:
            if trial_id == 1:
                key = subject_name + '_run'
            else:
                key = subject_name + '_run' + '_' + str(trial_id)
        
            
            # import dataset
            data = pd.read_csv(dir_data_overground + key + '.csv')
            # create two datasets, for two directions ran
            p = 0
            for i in range(0, len(data)):
                if data.loc[i,'usabledata'] == True and p == 0:
                    data_1_start = i
                    p = 1
                
                if data.loc[i,'usabledata'] == False and p == 1:
                    data_1_end = i
                    p = 2
                
                if data.loc[i,'usabledata'] == True and p == 2:
                    data_2_start = i
                    p = 3
                    
                if data.loc[i,'usabledata'] == False and p == 3:
                    data_2_end = i
                    break
             
            # from now on, overground data has 3 dimensions too. the third are the two directions.     
            data_split = list()    
            data_split.append(data[data_1_start:data_1_end])
            data_split.append(data[data_2_start:data_2_end])
            
                
            data_key_trials.append(key)      
            data_cut_trials.append(data_split)
    
        data_overground_key.append(data_key_trials)
        data_overground_cut.append(data_cut_trials)
        
    return data_overground_cut, data_overground_key


"""cut data into windows of length time_steps and move window with step step"""
def get_treadmill_windows(X_variables, y_variable, data_treadmill_cut, time_steps, step):
    ## data to use
    windows_treadmill_X = list()
    windows_treadmill_y = list()
    for subject_id in range(1,len(data_treadmill_cut)):
        windows_treadmill_X_subject = np.empty([0, time_steps, len(X_variables)])
        windows_treadmill_y_subject = np.empty([0, 1])
        for trial_id in range(0, len(data_treadmill_cut[subject_id])):
            
            for speed in range(0, len(data_treadmill_cut[subject_id][trial_id])):
                data_treadmill_X = data_treadmill_cut[subject_id][trial_id][speed].loc[:, X_variables]
                data_treadmill_y = data_treadmill_cut[subject_id][trial_id][speed].loc[:, y_variable]
                
                X, y =  \
                        create_dataset(data_treadmill_X, data_treadmill_y, \
                                           time_steps=time_steps, step=step)  
                
                            
                windows_treadmill_X_subject = np.append(windows_treadmill_X_subject, X, axis = 0)
                windows_treadmill_y_subject = np.append(windows_treadmill_y_subject, y, axis = 0)
        windows_treadmill_X.append(windows_treadmill_X_subject)
        windows_treadmill_y.append(windows_treadmill_y_subject)
    return windows_treadmill_X, windows_treadmill_y

"""cut data into windows of length time_steps and move window with step step"""
def get_overground_windows(X_variables, y_variable, data_overground_cut, time_steps, step):
    windows_overground_X = list()
    windows_overground_y = list()
    for subject_id in range(1,len(data_overground_cut)):
        windows_overground_X_subject = np.empty([0, time_steps, len(X_variables)])
        windows_overground_y_subject = np.empty([0, 1])
        for trial_id in range(0, len(data_overground_cut[subject_id])):
            for direction in range(0, len(data_overground_cut[subject_id][trial_id])):
                data_overground_X = data_overground_cut[subject_id][trial_id][direction].loc[:, X_variables]
                data_overground_y = data_overground_cut[subject_id][trial_id][direction].loc[:, y_variable]
               
                X, y =  \
                        create_dataset(data_overground_X, data_overground_y, \
                                           time_steps=time_steps, step=step)  
                            
                windows_overground_X_subject = np.append(windows_overground_X_subject, X, axis = 0)
                windows_overground_y_subject = np.append(windows_overground_y_subject, y, axis = 0)
        windows_overground_X.append(windows_overground_X_subject)
        windows_overground_y.append(windows_overground_y_subject)
    return windows_overground_X, windows_overground_y


def get_features_dataset(Xy,index):
    """from all features in one file to a collected features tensor and a label tensor"""
    """
    my_feature_0 = Xy[index,:,0]
    my_feature_0 = my_feature_0.astype(float)
    my_feature_1 = Xy[index,:,1]
    my_feature_1 = my_feature_1.astype(float)
    my_feature_2 = Xy[index,:,2]
    my_feature_2 = my_feature_2.astype(float)
    my_feature_3 = Xy[index,:,3]
    my_feature_3 = my_feature_3.astype(float)
    my_feature_4 = Xy[index,:,4]
    my_feature_4 = my_feature_4.astype(float)
    my_feature_5 = Xy[index,:,5]
    my_feature_5 = my_feature_5.astype(float)
    my_feature_6 = Xy[index,:,6]
    my_feature_6 = my_feature_6.astype(float)
    my_feature_7 = Xy[index,:,7]
    my_feature_7 = my_feature_7.astype(float)
    my_feature_8 = Xy[index,:,8]
    my_feature_8 = my_feature_8.astype(float)
    my_feature_9 = Xy[index,:,9]
    my_feature_9 = my_feature_9.astype(float)
    my_feature_10 = Xy[index,:,10]
    my_feature_10 = my_feature_10.astype(float)
    my_feature_11 = Xy[index,:,11]
    my_feature_11 = my_feature_11.astype(float)
    
    # subject id
    my_feature_12 = Xy[index,:,12]
    my_feature_12 = my_feature_12.astype(np.int64) # for int64
    # trial id
    my_feature_13 = Xy[index,:,13]
    my_feature_13 = my_feature_13.astype(np.int64) # for int64
    # speed
    my_feature_14 = Xy[index,:,14]
    my_feature_14 = my_feature_14.astype(float)
    # treadmill or overground
    my_feature_15 = Xy[index,:,15]
    my_feature_15 = my_feature_15.astype(np.int64) # for int64
    
    label = Xy[index,0,16]
    
    # put in one feature
    my_features = np.array([my_feature_0,my_feature_1, my_feature_2, my_feature_3, \
                            my_feature_4,my_feature_5,my_feature_6,my_feature_7,   \
                            my_feature_8,my_feature_9,my_feature_10,my_feature_11],\
                           np.float)
    my_features = np.transposfe(my_features)
    """
    
    #make a tensor
    features = tf.constant(Xy[index,:,0:12], dtype=tf.float32) 
    label = tf.constant(Xy[index,0,16], dtype=tf.int8)
    """
    # create one hot encoded array
    label = tf.keras.utils.to_categorical(
        label, num_classes = 112, dtype = 'int8'
        )
    """
    
    # convert to tensor file
   # features_dataset = tf.data.Dataset.from_tensor_slices((my_feature_4,my_feature_5,my_feature_6,my_feature_7,my_feature_8,my_feature_9,my_feature_10,my_feature_11,my_feature_12,my_feature_13,my_feature_14,my_feature_15,my_feature_16,my_feature_17,my_feature_18,my_feature_19,my_feature_20,my_feature_21,my_feature_22,my_feature_23,my_feature_24,my_feature_25,my_feature_26,my_feature_27,my_feature_28,my_feature_29,my_feature_30,my_feature_31,my_feature_32,my_feature_33,my_feature_34,my_feature_35,my_feature_36,my_feature_37,my_feature_38,my_feature_39,my_feature_40,my_feature_41,my_feature_45,my_feature_46,my_feature_47,my_feature_48,my_feature_49,my_feature_50,my_feature_51,my_feature_52,my_feature_53,my_feature_54, my_feature_72))
    return features, label

def save_to_tfrecord(my_features, label, tfrecord_dir):
    # needs features as a tensor and label as a single number or a hot encoded array (int)
    with tf.io.TFRecordWriter(tfrecord_dir) as writer:
        example = make_example(my_features, label)
        writer.write(example)

def make_example(my_features, label):
    
    features_feature = Feature(
        bytes_list=BytesList(value=[
            tf.io.serialize_tensor(my_features).numpy(),
        ])
    )
    label_feature = Feature(
        bytes_list=BytesList(value=[
            tf.io.serialize_tensor(label).numpy(),
        ])
    )

    features = Features(feature={
        'features': features_feature,
        'label': label_feature,
    })
    
    example = Example(features=features)
    
    return example.SerializeToString()

def read_tfrecord_running(serialized_example):
    # How to:
    # tfrecord_dataset = tf.data.TFRecordDataset([list with file names])
    # parsed_dataset = tfrecord_dataset.map(read_tfrecord)
    feature_description = {
        'features': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    features = tf.io.parse_tensor(example['features'], out_type = tf.float32)
    features.set_shape([5000, 12]) 
    label = tf.io.parse_tensor(example['label'], out_type = tf.int8)
    #label = label[0:3]
    label.set_shape(1) # 112
    
    return features, label


