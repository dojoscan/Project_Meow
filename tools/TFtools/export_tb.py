import tensorflow as tf
import os

# WRITE SCALARS FROM TENSORBOARD LOG FILE TO TXT FILE

path_to_events_folder = '/Users/Donal/Dropbox/CIFAR10/meow_logs/squeeze_res/'
dir_files = os.listdir(path_to_events_folder)
output_filename = 'accuracy_100_it.txt'
output_file = open(path_to_events_folder+output_filename, 'w')

# Reads from last file in events folder
for e in tf.train.summary_iterator(path_to_events_folder + dir_files[-1]):
    for v in e.summary.value:
        if v.tag == 'Accuracy':
            val = v.simple_value
            output_file.write("%f \n" % val)