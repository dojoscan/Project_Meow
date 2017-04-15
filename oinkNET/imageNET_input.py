# INPUT PIPELINE FOR IMAGENET

import tensorflow as tf
import parameters as p
import os


def read_image(filename):
    """
    Args:
        filename: a scalar string tensor.
    Returns:
        image_tensor: decoded image
    """
    with tf.variable_scope('ReadImage'):
        file_contents = tf.read_file(filename)
        image = tf.image.decode_jpeg(file_contents, channels=3)
        image = tf.cast(image, tf.float32)
        with tf.variable_scope('DistortImage'):
            binary = tf.random_shuffle([0, 1])
            image = tf.image.random_flip_left_right(image)
            if binary[0] == 0:
                image = tf.image.random_brightness(image, max_delta=50. / 255.)
                image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
            else:
                image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
                image = tf.image.random_brightness(image, max_delta=50. / 255.)
        image = tf.image.resize_image_with_crop_or_pad(image, p.PRIM_IMAGE_HEIGHT, p.PRIM_IMAGE_WIDTH)
    return image


def read_labels(path_to_labels):
    """
    Args:
        path_to_labels: full path to labels file
    Returns:
        labels: a 1D tensor of all labels
    """

    f = open(path_to_labels, 'r')
    labels = []
    for line in f:
        labels.append(int(line))
    return labels


def create_image_list_val(path_to_images):
    """
    Args:
        path_to_images: full path to validation image folder
    Returns:
        image_list: a list of all files in that folder
    """

    image_list = os.listdir(path_to_images)
    image_list = [path_to_images + s for s in image_list]
    return image_list


def create_image_list_train(path_to_images):
    """
    Args:
        path_to_images: full path to training image folder
    Returns:
        image_list: a list of all files in that folder
    """

    image_list = []
    for path, subdirs, files in os.walk(path_to_images):
        for name in files:
            image_list.append(path + '/' + name)
    return image_list


def create_batch(batch_size, mode):
    """
    Args:
        path_to_images: full path to input images folder
        path_to_labels: full path to input labels
        batch_size: number of examples in mini-batch
        train: boolean for training or testing mode
    Returns:
        batch: list of images as a 4d tensor sz = [batch_sz, im_h, im_w, im_d]
        and labels as a 1d tensor sz = [batch_sz]
    """

    with tf.variable_scope('KITTIInputPipeline'):
        if mode == 'Train':
            path_to_images = p.PATH_TO_PRIM_DATA + 'training/image/'
            path_to_labels = p.PATH_TO_PRIM_DATA + 'training/labels.txt'
            image_list = create_image_list_train(path_to_images)
        else:
            path_to_images = p.PATH_TO_PRIM_DATA + 'validation/image/'
            path_to_labels = p.PATH_TO_PRIM_DATA + 'validation/labels.txt'
            image_list = create_image_list_val(path_to_images)
        labels = read_labels(path_to_labels)
        image_list = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        input_queue = tf.train.slice_input_producer([image_list, labels], shuffle=True,  name='InputProducer')
        images = read_image(input_queue[0])
        batch = tf.train.batch([images, input_queue[1]], batch_size=batch_size, name='Batch', num_threads=p.NUM_THREADS)
    return batch

