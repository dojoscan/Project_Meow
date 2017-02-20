import sys
sys.path.insert(0, '/Users/Donal/Desktop/Thesis/Code/Project_Meow/meowNET')
import input
from PIL import Image
import numpy as np
import tensorflow as tf
import os

PATH_TO_IMAGES = "/Users/Donal/Dropbox/CIFAR10/Data/train/images/"
PATH_TO_LABELS = "/Users/Donal/Dropbox/CIFAR10/Data/train/labels.txt"
PATH_TO_OUTPUT = '/Users/Donal/Desktop'
image_list = input.create_image_list(PATH_TO_IMAGES)
labels = input.read_labels(PATH_TO_LABELS)
num_examples = len(labels)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

filename = os.path.join(PATH_TO_OUTPUT + '/CIFAR10.tfrecords')
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    if index % 100 == 0:
        print('Converting image no. ' + str(index))
    img = np.array(Image.open(image_list[index]))
    height = img.shape[0]
    width = img.shape[1]
    img_raw = img.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw),
        'label': _int64_feature(int(labels[index]))}))
    writer.write(example.SerializeToString())
writer.close()
