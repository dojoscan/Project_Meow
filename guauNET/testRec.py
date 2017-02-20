import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tfrecords_filename = '/Users/Donal/Dropbox/CIFAR10/Data/train/CIFAR10.tfrecords'

reconstructed_images = []
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)

    height = int(example.features.feature['height']
                 .int64_list
                 .value[0])

    width = int(example.features.feature['width']
                .int64_list
                .value[0])

    img_string = (example.features.feature['image_raw']
                  .bytes_list
                  .value[0])

    annotation_string = (example.features.feature['label']
                         .int64_list
                         .value[0])

    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((height, width, -1))
    print(reconstructed_img)
    print(annotation_string)
