import tensorflow as tf
import os
import numpy as np
import time

from parameters import IMAGE_HEIGHT, IMAGE_WIDTH,CLASSES
import tools as t

def read_image(filename):
    """
    Args:
        filename: a scalar string tensor.
    Returns:
        image_tensor: decoded image
    """

    file_contents = tf.read_file(filename)
    image = tf.image.decode_png(file_contents, channels=3)
    image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return image

def read_labels(path_to_labels):
    """
    Args:
        path_to_labels: full path to labels file
    Returns:
        bbox: a 1D list of arrays of all boxes, array_sz=[number of times * 4]
        classes: a 1D list of array of all classes, array_ sz=[number of objects]
    """
    label_list = os.listdir(path_to_labels)
    bboxes = []
    classes = []
    index_images = []
    i=0
    for file in [path_to_labels + s for s in label_list]:
        data = open(file, 'r').read()
        # separate the values for each image, every image can have more than one object
        line = data.split()
        # an array with the index of each image
        index_images.append(i)
        for obj in range(0, len(line), 15):
            # extract ground truth coordinates [x,y,w,h] and class for each object
            annot = line[obj + 0]
            cls = int(CLASSES[annot]) # CLASSES convert classes to a number, eg. 'Pedestrian' to 3
            xmin = float(line[4 + obj])
            ymin = float(line[5 + obj])
            xmax = float(line[6 + obj])
            ymax = float(line[7 + obj])
            x, y, w, h = t.bbox_transform_inv([xmin, ymin, xmax, ymax])
            # create a new classes and bboxes lists, if first object create new list
            if obj == 0:
                classes.append([cls])
                bboxes.append([(x,y,w,h)])
            else:
                ind=index_images.index(i)
                classes[ind].append(cls)
                bboxes[ind].append((x,y,w,h))
        i=i+1
    return classes, bboxes


def create_image_list(path_to_images):
    """
    Args:
        path_to_images: full path to image folder
    Returns:
        image_list: a tensor of all files in that folder
    """

    image_list = os.listdir(path_to_images)
    image_list = [path_to_images + s for s in image_list]
    return image_list

def create_batch(path_to_images, path_to_labels, batch_size, train):
    """
    Args:
        path_to_images: full path to input images folder
        path_to_labels: full path to input labels folder
        batch_size: number of examples in mini-batch
        train: boolean for training or testing mode
    Returns:
        batch: list of images as a 4d tensor sz = [batch_sz, im_h, im_w, im_d]
        and labels as a 1d tensor sz = [batch_sz]
    """
    image_list = create_image_list(path_to_images)
    no_samples = len(image_list)
    image_list = tf.convert_to_tensor(image_list, dtype=tf.string)
    if train:
       classes, bboxes = read_labels(path_to_labels)
    else:   # Create fake labels for testing data
        classes = [0]*no_samples
        bboxes=[0]*no_samples
    classes = tf.convert_to_tensor(classes, dtype=tf.int32)
    bboxes=tf.convert_to_tensor(bboxes, dtype=tf.float32)
    input_queue = tf.train.slice_input_producer([image_list, classes, bboxes], shuffle=False)
    images = read_image(input_queue[0])
    batch = tf.train.batch([images, input_queue[1], input_queue[2]], batch_size=batch_size)
    return batch

