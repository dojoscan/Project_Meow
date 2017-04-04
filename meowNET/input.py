import tensorflow as tf
import os

IM_SIZE = 32

def read_image(filename):
    """
    Args:
        filename: a scalar string tensor.
    Returns:
        image_tensor: decoded image
    """
    with tf.variable_scope('ReadImage'):
        file_contents = tf.read_file(filename)
        image = tf.image.decode_png(file_contents, channels=3)
        image = tf.image.resize_images(image, [IM_SIZE, IM_SIZE])
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
        path_to_labels: full path to input labels
        batch_size: number of examples in mini-batch
        train: boolean for training or testing mode
    Returns:
        batch: list of images as a 4d tensor sz = [batch_sz, im_h, im_w, im_d]
        and labels as a 1d tensor sz = [batch_sz]
    """
    image_list = create_image_list(path_to_images)
    no_samples = len(image_list)
    image_list = tf.convert_to_tensor(image_list, dtype=tf.string, name='ImageList')
    if train:
        labels = read_labels(path_to_labels)
    else:   # Create fake labels for testing data
        labels = [0]*no_samples
    labels = tf.convert_to_tensor(labels, dtype=tf.int32, name='Labels')
    input_queue = tf.train.slice_input_producer([image_list, labels], shuffle=True, name='InputQueue')
    images = read_image(input_queue[0])
    # Dequeue mini-batch
    batch = tf.train.batch([images, input_queue[1]], batch_size=batch_size, name='Batch')
    return batch

