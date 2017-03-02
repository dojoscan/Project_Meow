import tensorflow as tf
import os
import numpy as np

from parameters import IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES
from loss import bbox_transform_inv

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
        labels: a 1D tensor of all labels
        bbox:
    """
    label_list = os.listdir(path_to_labels)
    labels = []
    for file in [path_to_labels + s for s in label_list]:
        data = open(file, 'r').read()
        labels.append(data)

    num_obj=[]
    for i in range(0,len(labels)):
        line=labels[i].split()
        num_obj.append(int(len(line)/15))

    max_num_obj=max(num_obj)
    separacion = np.zeros((len(labels), max_num_obj,1))
    print(separacion)
    for i in range(0, len(labels)):
        # separate the values for each image, every image can have more than one object
        line = labels[i].split()
        bboxes =[]
        for obj in range(0, len(line), 15):
            # for each object in the image, we extract the ground truth corrdinates [x,y,w,h] and the class of the object
            annot = line[obj + 0]
            cls = int(CLASSES[annot])
            xmin = float(line[4 + obj])
            ymin = float(line[5 + obj])
            xmax = float(line[6 + obj])
            ymax = float(line[7 + obj])
            x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
            bboxes.append([x, y, w, h, cls])
        print(bboxes)
        separacion[i]=bboxes
    print(separacion)

        #labels[index] = bboxes
        #index = index + 1

    return labels


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
        labels = read_labels(path_to_labels)
    else:   # Create fake labels for testing data
        labels = [0]*no_samples
    labels = tf.convert_to_tensor(labels, dtype=tf.string)
    input_queue = tf.train.slice_input_producer([image_list, labels], shuffle=False)
    images = read_image(input_queue[0])
    batch = tf.train.batch([images, input_queue[1]], batch_size=batch_size)
    return batch

