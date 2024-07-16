#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:12:25 2023

@author: patrick
"""
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve, PrecisionRecallDisplay, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
AUTOTUNE = tf.data.AUTOTUNE
import keras
from tensorflow.keras.metrics import top_k_categorical_accuracy
import random
from functools import reduce
from tensorflow.keras import backend as K


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

def compile_model_callbacks_etc(filepath, output_variable, early_stopping_min_delta, 
                                early_stopping_patience, reinitialize_epochs, model,
                                flag_top_5_accuracy, learning_rate):
    
    if learning_rate:
        optimizer = keras.optimizers.Adam(learning_rate = learning_rate)
        print('optimizer with new learning rate')
    else:
        print('optimizer with default learning rate')
        optimizer = keras.optimizers.Adam()
    
    
    if output_variable == 'speed' or output_variable == 'subject_id' or output_variable == 'bin_label':
        
        monitor = 'val_accuracy'
        
        my_metrics = ["accuracy", 
                   #tf.keras.metrics.AUC(curve = 'ROC'),
                   #tf.keras.metrics.AUC(curve = 'PR'),
                   #tf.keras.metrics.Precision(),
                   #tf.keras.metrics.Recall(),
                   #tf.keras.metrics.PrecisionAtRecall(0.8) 
                  ] #tf.keras.metrics.AUC(from_logits=True)
        loss_to_use = 'categorical_crossentropy'
        
        if flag_top_5_accuracy:
            monitor = 'val_accuracy'
            my_metrics= ['accuracy', top_5_accuracy
                       #tf.keras.metrics.AUC(curve = 'ROC'),
                       #tf.keras.metrics.AUC(curve = 'PR'),
                       #tf.keras.metrics.Precision(),
                       #tf.keras.metrics.Recall(),
                       #tf.keras.metrics.PrecisionAtRecall(0.8) 
                       #tf.keras.metrics.AUC(from_logits=True)\
                           ]
            loss_to_use = 'categorical_crossentropy'
            
        
    elif output_variable == 'seconds_10k':
        

        monitor = 'val_mean_squared_error'
       
        
        my_metrics = ['mean_squared_error',
                   #tf.keras.metrics.AUC(curve = 'ROC'),
                   #tf.keras.metrics.AUC(curve = 'PR'),
                   #tf.keras.metrics.Precision(),
                   #tf.keras.metrics.Recall(),
                   #tf.keras.metrics.PrecisionAtRecall(0.8) 
                  ] #tf.keras.metrics.AUC(from_logits=True)
        loss_to_use = 'mean_squared_error'
        
    else:
        print('output variable not specified correctly')
        
    "callbacks"
    print('model checkpoint included')
    check_point = keras.callbacks.ModelCheckpoint(filepath,
                                                 verbose = 1,
                                                 monitor=monitor,
                                                 save_best_only=True,
                                                 mode='auto', # if we save_best_only, we need to specify on what rule. Rule here is if val_loss is minimum, it owerwrites
                                                 #save_weights_only = True,  # to only save weights, otherwise it will save whole model
                                                 )
    
    
    
    # make sure to add this to the fit model again when uncommenting
    print('early stopping included') 
    earlystopping = tf.keras.callbacks.EarlyStopping( 
                    monitor=monitor,
                    min_delta=early_stopping_min_delta,
                    patience=early_stopping_patience,
                    verbose=1,
                    mode='auto',
                    restore_best_weights=True # Whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used.
                    )
    
    
    reinitialize_callback = ReinitializeWeightsCallback(reinitialize_epochs=reinitialize_epochs, metric_type=monitor)
    
    my_callbacks = [earlystopping, check_point, reinitialize_callback]
    
    
    model.compile(loss = loss_to_use, 
                  optimizer = optimizer, 
                  metrics = my_metrics) #tf.keras.metrics.AUC(from_logits=True)
    
    
    
    return model, my_callbacks

def parse_spectrogram_tfrecord(example):
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
  
  example["height"] = example['height']
  example["width"] = example['width']
  example["depth"] = example['depth']
  example["score_10k"] = example['score_10k']
  example["seconds_10k"] = example['seconds_10k']
  example["spectrogram_image"] = tf.io.parse_tensor(example['spectrogram_image'], out_type=tf.double)
  example["spectrogram_image"] = tf.reshape(example["spectrogram_image"], shape=[example["height"],example["width"],example["depth"]])
  example["bin_label"] = example['bin_label']
  example["bin_label"] = tf.one_hot(example["bin_label"], depth = 2, dtype = 'int64')
  example["subject_id"] = example['subject_id']-1
  example["subject_id"] = tf.one_hot(example["subject_id"], depth = 188, dtype = 'int64')
  example["speed"] = example['speed_label']
  example["speed"] = tf.one_hot(example["speed"], depth = 4, dtype = 'int64')
  
  return example


# not needed anymore?
def parse_raw_tfrecord(example, window_length):
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
    example["bin_label"] = example["bin_label"]
    example["seconds_10k"] = example["seconds_10k"]
    example["subject_id"] = example["subject_id"]
    example["tread_or_overground"] = example["tread_or_overground"]
    example["speed"] = example["speed"]
    example["speed_onehot"] = tf.sparse.to_dense(example["speed_onehot"])
    example["subject_id_onehot"] = tf.sparse.to_dense(example["subject_id_onehot"])
    
    #example["speed"] = tf.one_hot(example['speed']-1, depth = 3, dtype = 'int64')
    return example

def prepare_spectrogram_sample(features, output_variable):
    spectrogram_image = features["spectrogram_image"]
    output = features[output_variable]
    
    return spectrogram_image, output

def prepare_bin_and_seconds_sample(features):
    bin_label = features["bin_label"]
    seconds_10k = features["seconds_10k"]
    
    return bin_label, seconds_10k
    
   
# can delete this
def prepare_spectrogram_sample_speed(features):
  spectrogram_image = features["spectrogram_image"]
  speed = features["speed"]
  return spectrogram_image, speed  


# can delete this
def prepare_spectrogram_sample_subjectid(features):
  spectrogram_image = features["spectrogram_image"]
  subject_id = features["subject_id"]
  return spectrogram_image, subject_id   

"""
def get_seconds_bin(filenames, batch_size, output_variable):
    dataset = (
            tf.data.TFRecordDataset(filenames, num_parallel_reads= None)
            .map(parse_spectrogram_tfrecord)
            .map(lambda x: prepare_spectrogram_sample(x, output_variable), num_parallel_calls=None)
            #.shuffle(batch_size*10)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))
            
    return dataset
"""

def prepare_raw_sample_multipleinputs_speed(features):
    #image = tf.image.resize(features["image"], size=(224, 224))
    #return image, features["category_id"]
    left_ax = features["left_ax"]
    left_ay = features["left_ay"]
    left_az = features["left_az"]
    left_gx = features["left_gx"]
    left_gy = features["left_gy"]
    left_gz = features["left_gz"]
    right_ax = features["right_ax"]
    right_ay = features["right_ay"]
    right_az = features["right_az"]
    right_gx = features["right_gx"]
    right_gy = features["right_gy"]
    right_gz = features["right_gz"]
    speed_onehot = features['speed_onehot']
    
    return (left_ax, left_ay, left_az, left_gx, left_gy, left_gz, right_ax, right_ay, right_az, right_gx, right_gy, right_gz), speed_onehot #, left_az, left_gx, left_gy, left_gz, right_ax, right_ay, right_az, right_gx, right_gy, right_gz

def prepare_raw_sample_singleinput_speed(features):
    #image = tf.image.resize(features["image"], size=(224, 224))
    #return image, features["category_id"]
    left_ax = features["left_ax"]
    left_ay = features["left_ay"]
    left_az = features["left_az"]
    left_gx = features["left_gx"]
    left_gy = features["left_gy"]
    left_gz = features["left_gz"]
    right_ax = features["right_ax"]
    right_ay = features["right_ay"]
    right_az = features["right_az"]
    right_gx = features["right_gx"]
    right_gy = features["right_gy"]
    right_gz = features["right_gz"]
    speed_onehot = features['speed_onehot']
    
    return left_ax, speed_onehot

# filters out percentage of data. 100% uses all data, everything under is less data than what we have got
def filter_percentage(counter, percentage):
    return (counter % 100) < percentage 

def get_spectrogram_dataset(filenames, batch_size, output_variable, percentage_of_data_to_be_used):
    # Set the random seed for reproducibility
    random.seed(42)

    dataset_list = []

    for file in filenames:
        # Create the dataset for each file individually
        dataset = tf.data.TFRecordDataset(file)
        dataset = dataset.map(parse_spectrogram_tfrecord)

        if percentage_of_data_to_be_used < 100:  # Apply the percentage filter if the percentage is less than 100
            dataset = dataset.enumerate()
            dataset = dataset.filter(lambda index, _: filter_percentage(index, percentage_of_data_to_be_used))
            dataset = dataset.map(lambda _, x: x)  # Remove the index from the examples

        dataset_list.append(dataset)

    # Use reduce to sequentially concatenate all the datasets in the list
    dataset = reduce(lambda ds1, ds2: ds1.concatenate(ds2), dataset_list)

    dataset = dataset.map(lambda x: prepare_spectrogram_sample(x, output_variable), num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
    

def get_spectrogram_dataset_back_to_the_roots(filenames, batch_size, output_variable):    
    #create the dataset
    dataset = (
            tf.data.TFRecordDataset(filenames, num_parallel_reads= None)
            .map(parse_spectrogram_tfrecord)
            .map(lambda x: prepare_spectrogram_sample(x, output_variable), num_parallel_calls=AUTOTUNE)
            .shuffle(batch_size*10)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
            )


            
    return dataset

def get_partial_dataset(filenames, batch_size, output_variable, percentage_of_data_to_be_used):
    if percentage_of_data_to_be_used == 1:
        final_dataset = (
                tf.data.TFRecordDataset(filenames, num_parallel_reads= None)
                .map(parse_spectrogram_tfrecord)
                .map(lambda x: prepare_spectrogram_sample(x, output_variable), num_parallel_calls=AUTOTUNE)
                .shuffle(batch_size*10)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
                )
        
    else:
        partial_dataset = []
        seed = 123
        for filename in filenames:
            # Create a dataset from the TFRecord file
            dataset = (
                tf.data.TFRecordDataset(filename, num_parallel_reads=None)
                .map(parse_spectrogram_tfrecord)
                .map(lambda x: prepare_spectrogram_sample(x, output_variable), num_parallel_calls=tf.data.AUTOTUNE)
            )
            
    
            # Count the total number of instances in the TFRecord file
            num_instances = sum(1 for _ in tf.data.TFRecordDataset(filename))
            #print("Number of instances in the training dataset:", num_instances)
            
            # Calculate the number of instances to keep based on the specified percentage
            num_partial_instances = int(num_instances * percentage_of_data_to_be_used)
            
            # Set the random seed for reproducibility
            random.seed(seed)
            
            #take random instances frmo the dataset
            instances = random.sample(list(dataset.as_numpy_iterator()), num_partial_instances)
            
            #create a dataset with the instances
            for instance in instances:
                partial_dataset.append(instance)
          
        
        partial_batched_dataset = tf.data.Dataset.from_generator(lambda: partial_dataset, output_signature=(
            tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64)
        ))
        partial_batched_dataset = partial_batched_dataset.shuffle(buffer_size=len(partial_dataset)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        final_dataset = partial_batched_dataset
        
    return final_dataset

def get_bin_and_seconds(filenames, batch_size):
    #create the dataset
    dataset = (
            tf.data.TFRecordDataset(filenames, num_parallel_reads= None)
            .map(parse_spectrogram_tfrecord)
            .map(prepare_bin_and_seconds_sample, num_parallel_calls=None)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))
            
    return dataset




def performance_test_dataset(model, filenames, batch_size, final_layer_size, model_to_use, output_variable, window_length, mean_seconds_training,percentage_of_data_to_be_used):
    
    # this can go
    if output_variable == "speed":
        test_dataset = get_spectrogram_dataset(filenames, batch_size, output_variable, percentage_of_data_to_be_used)
            
    elif output_variable == "subject_id" or output_variable == "seconds_10k" or output_variable == "bin_label":
        #test_dataset = get_spectrogram_dataset(filenames, batch_size, output_variable, percentage_of_data_to_be_used)
        
        # for testing original function (without reducing dataset)
        #test_dataset = get_spectrogram_dataset_back_to_the_roots(filenames, batch_size, output_variable)
        
        # new partial dataset import function
        test_dataset = get_partial_dataset(filenames, batch_size, output_variable, percentage_of_data_to_be_used)
        
        
        
    acc = model.evaluate(test_dataset, steps = len(filenames))
    pred = model.predict(test_dataset)
    print('Test_accuracy: ' + str(acc))
    
    
    if output_variable == 'speed' or output_variable == 'subject_id' or output_variable == 'bin_label':
        steps_to_take = len(filenames)
        
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
        output_function = [true_list, pred_list]
        
    else:
        steps_to_take = len(filenames)
        true_list = []
        pred_values_list = []
        for x, y in test_dataset.take(steps_to_take):
            
            true_list = true_list + list(y.numpy().astype(int))
            pred_value = model.predict(x)
            pred_values_list = pred_values_list + list(pred_value)
         
            
        mean_seconds_same_length =  np.full((len(true_list),), mean_seconds_training) 
        
        print('mean_seconds_from_training: ' + str(mean_seconds_training)) 
        print('manual loss from mean')
        print(mean_squared_error(true_list, mean_seconds_same_length))
        print('manual loss from model: ')
        print(mean_squared_error(true_list, pred_values_list))
        
        output_function = [acc, pred]
    
    return output_function

 
def get_filenames_seconds(subjects, trials, dir_tfr, flag_shuffle_train, test_subjects, val_split):
 
    
    val_subjects = subjects[0:int(len(subjects)*val_split)]
    train_subjects = subjects[int(len(subjects)*val_split):len(subjects)]
    
    train_filenames = list()
    val_filenames = list() 
    test_filenames = list()
    
    speeds = [0,1,2]
    
    
    for subject in test_subjects + subjects:
        sensor = "SENSOR" + "{:03d}".format(subject)
        for trial in trials:
            if trial == 1:
                for speed in speeds:
                    dir_tfr_data = dir_tfr + 'Treadmill/' + 'speed' + str(speed) + '/' + sensor + ".tfrecords"
                    if subject in test_subjects:
                        test_filenames.append(dir_tfr_data)
                    elif subject in val_subjects:
                        val_filenames.append(dir_tfr_data)
                    else:
                        train_filenames.append(dir_tfr_data)
                    
            else:
                for speed in speeds:
                    dir_tfr_data = dir_tfr + 'Treadmill/' + 'speed' + str(speed) + '/' + sensor + '_' + str(trial) + ".tfrecords"
                    if subject in test_subjects:
                        test_filenames.append(dir_tfr_data)
                    elif subject in val_subjects:
                        val_filenames.append(dir_tfr_data)
                    else:
                        train_filenames.append(dir_tfr_data)
    
    
    
    if flag_shuffle_train:
        np.random.shuffle(train_filenames)    
        
    return train_filenames, val_filenames, test_filenames    
 
def get_filenames_subjectid_classification(subjects, trials, dir_tfr, flag_shuffle_train, test_subjects):
    
    train_filenames = list()
    val_filenames = list()
    test_filenames = list()
    
    speeds = [0,1,2]
    
    
    for subject in subjects:
        sensor = "SENSOR" + "{:03d}".format(subject)
        for trial in trials:
            if trial == 1:
                for speed in speeds:
                    dir_tfr_data = dir_tfr + 'Treadmill/' + 'speed' + str(speed) + '/' + sensor + ".tfrecords"
                    train_filenames.append(dir_tfr_data)
                dir_tfr_data = dir_tfr + 'Overground/' + sensor + ".tfrecords"
                if subject in test_subjects:
                    test_filenames.append(dir_tfr_data)
                else:
                    val_filenames.append(dir_tfr_data)
            else:
                for speed in speeds:
                    dir_tfr_data = dir_tfr + 'Treadmill/' + 'speed' + str(speed) + '/' + sensor + '_' + str(trial) + ".tfrecords"
                    train_filenames.append(dir_tfr_data)
                dir_tfr_data = dir_tfr + 'Overground/' + sensor + '_' + str(trial) + ".tfrecords"
                if subject in test_subjects:
                    test_filenames.append(dir_tfr_data)
                else:
                    val_filenames.append(dir_tfr_data)
    
    
    if flag_shuffle_train:
        np.random.shuffle(train_filenames)    
        
    return train_filenames, val_filenames, test_filenames
    
def get_filenames_speed_classification(subjects, test_split, val_split, dir_tfrecords, flag_shuffle_train):
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
    
    return train_filenames, val_filenames, test_filenames


def plot_accelerations(tense):
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
   
        
class ReinitializeWeightsCallback(tf.keras.callbacks.Callback):
    def __init__(self, reinitialize_epochs, metric_type):
        super(ReinitializeWeightsCallback, self).__init__()
        self.reinitialize_epochs = reinitialize_epochs
        self.metric_type = metric_type
        self.last_reinitialize_epoch = 0
        self.last_val_metric = -np.Inf
        self.best_val_metric = -np.Inf
        self.highest_val_metric_since_last_reinit = -np.Inf
        self.initial_weights = None

    def on_epoch_end(self, epoch, logs=None):
        if self.metric_type == 'val_accuracy':
            current_val_metric = logs.get('val_accuracy')
        elif self.metric_type == 'val_mean_squared_error':
            current_val_metric = -logs.get('val_mean_squared_error')
        else:
            raise ValueError('Invalid metric type: {}'.format(self.metric_type))

        if current_val_metric is None:
            return

        if current_val_metric > self.best_val_metric:
            self.best_val_metric = current_val_metric
            self.last_val_metric = epoch
            self.highest_val_metric_since_last_reinit = current_val_metric
            self.best_weights = self.model.get_weights()
        elif epoch - self.last_val_metric > self.reinitialize_epochs and epoch > 0:
            print('Reinitializing weights at epoch', epoch)
            self.model.set_weights(self.initial_weights)
            self.last_reinitialize_epoch = epoch
            self.last_val_metric = epoch
            self.best_val_metric = self.highest_val_metric_since_last_reinit
        elif current_val_metric > self.highest_val_metric_since_last_reinit:
            self.highest_val_metric_since_last_reinit = current_val_metric

        if epoch == 0:
            self.initial_weights = self.model.get_weights()


    