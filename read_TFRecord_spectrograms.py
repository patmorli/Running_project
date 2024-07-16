# read_TFRecord_spectrograms.py, Patrick Mayerhofer, July 16
# this script parses the TFRecord files, and plots the spectrogram

import tensorflow as tf
import functions_classification as fc
import matplotlib.pyplot as plt
import IPython

# this is the directory and filename of the TFRecord
dir_file = '/Users/patmorli/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@weartechlabs.com/My Drive/Running Plantiga Project - Backup From Locomotion/Data/Prepared/tfrecords/10k_1000_250_0/Treadmill/speed1/SENSOR058.tfrecords'

# this is the output variable that we want to predict
output_variable = "bin_label" #"speed", "subject_id", "seconds_10k", "bin_label"

# to 
dataset = (
                tf.data.TFRecordDataset(dir_file, num_parallel_reads=None)
                .map(fc.parse_spectrogram_tfrecord)
                .map(lambda x: fc.prepare_spectrogram_sample(x, output_variable), num_parallel_calls=tf.data.AUTOTUNE)
            )

# create a TensorFlow dataset from a set of TFRecord files. 
# this does not load the data, but only creates a dataset object that can be used to load the data later, to save memory
dataset = (
    tf.data.TFRecordDataset(dir_file, num_parallel_reads=None)
    .map(fc.parse_spectrogram_tfrecord)
    .map(lambda x: fc.prepare_spectrogram_sample(x, output_variable), num_parallel_calls=tf.data.AUTOTUNE)
)

# load all data from the dataset into a list -> typically, this will be done durring training or testing
list_dataset = list(dataset)

# get one single spectrogram from the list
# the first index [0] is the sample, the second index [0] is the spectrogram data (model input), [:,:,0] is the first channel of the IMU data
spectrogram_data = list_dataset[0][0][:,:,0] 

# plot the spectrogram
plt.imshow(spectrogram_data, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Intensity')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram')
#plt.show


# Save the plot to a file instead of displaying it here (because for some reason plt.show() does not work in the notebook)
plt.savefig('spectrogram.png') # could add a specific path

