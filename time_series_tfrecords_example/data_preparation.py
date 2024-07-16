#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 18:59:59 2022

@author: patrickmayerhofer

data_preparation

https://www.kaggle.com/code/danmaccagnola/activity-recognition-data-w-tfrecords/notebook
"""



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import glob
import os
np.random.seed(1111)

from pathlib import Path 

import tensorflow as tf
from tensorflow.keras import layers, models 
from keras.utils.vis_utils import plot_model

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve, PrecisionRecallDisplay

target = 'running'
filename = f'{target}-1'
path = f"/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Running Plantiga Project/Code/time_series_tfrecords_example/data/{target}/{filename}.csv"
df_example = pd.read_csv(path)
df_example = df_example.iloc[:,0:3]

list_samples_example = df_example.to_dict('records')

dict_samples_example = {
    'filename': filename,
    'fullpath': path,
    'target' : target,
    'samples': list_samples_example
} 


files = np.sort(glob.glob("/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Running Plantiga Project/Code/time_series_tfrecords_example/data/*/*"))
np.random.shuffle(files)


output_list = []

for file in files[0:200]:
    print(file, end='\r')
    parts = Path(file).parts
    target = parts[-2]
    filename = parts[-1]
    
    df_cur = pd.read_csv(file)
    df_cur = df_cur.iloc[:,0:3]
    
    list_samples_cur = df_cur.to_dict('records')
    
    dict_samples_cur = {
        'filename': filename,
        'fullpath': file,
        'target' : target,
        'samples': list_samples_cur
    } 
    
    output_list.append(dict_samples_cur)


output_list_df = pd.DataFrame(output_list)


output_list_df.to_parquet("/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Running Plantiga Project/Code/time_series_tfrecords_example/working/parquet_data.parquet")


"""create TFRecords"""
full_dataset_parquet = pd.read_parquet("/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Running Plantiga Project/Code/time_series_tfrecords_example/working/parquet_data.parquet") 

mapping = {item:i for i, item in enumerate(full_dataset_parquet['target'].unique())}
full_dataset_parquet["target_id"] = full_dataset_parquet["target"].apply(lambda x: mapping[x])   



full_dataset_parquet['id'] = full_dataset_parquet['filename'].str.extract("\w+-(\d+).csv").astype(int)

full_dataset_parquet['unique_id'] = full_dataset_parquet['id'] + ((1+full_dataset_parquet["target_id"])*1e7).astype(int)

#full_dataset_parquet
#full_dataset_parquet['samples'][2]

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(example):
    feature = {
        "acc_x": float_feature_list([x['accelerometer_X'] for x in example['samples']]),
        "acc_y": float_feature_list([x['accelerometer_Y'] for x in example['samples']]),
        "acc_z": float_feature_list([x['accelerometer_Z'] for x in example['samples']]),
        "acc_matrix": float_feature_list(
            pd.DataFrame.from_records([x for x in example['samples']]).values.reshape(-1)
        ),
        "target_id": int64_feature(example["target_id"]),
        "fullpath": bytes_feature(example["fullpath"]),
        "id": int64_feature(example['unique_id']),
        "target_id_onehot": float_feature_list(tf.keras.utils.to_categorical(example["target_id"], 
                                                                        num_classes=len(mapping.keys())
                                                                       )
                                         )
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

num_samples = 128
num_tfrecords = full_dataset_parquet.shape[0] // num_samples
if full_dataset_parquet.shape[0] % num_samples:
    num_tfrecords += 1  # add one record if there are any remaining samples

print(num_tfrecords)

tfrecords_dir = '/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Running Plantiga Project/Code/time_series_tfrecords_example/working/tfrecords'

if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)  # creating TFRecords output folder
    

for tfrec_num in range(num_tfrecords):
    samples = full_dataset_parquet.loc[(tfrec_num * num_samples) : ((tfrec_num + 1) * num_samples)]

    with tf.io.TFRecordWriter(
        tfrecords_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples)-1)
    ) as writer:
        for index, row_sample in samples.iterrows():
            example = create_example(row_sample)
            writer.write(example.SerializeToString())
            
            