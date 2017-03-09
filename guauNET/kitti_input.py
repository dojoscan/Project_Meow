# READS BATCHS OF IMAGES AND LABELS (MASKS, COORDS, DELTAS, CLASSES) FROM DISK

import tensorflow as tf
import os
import parameters as p

def read_image(filename):
    """
    Args:
        filename: a scalar string tensor.
    Returns:
        image_tensor: decoded image
    """

    file_contents = tf.read_file(filename)
    image = tf.image.decode_png(file_contents, channels=3)
    if p.DATA_AUGMENT:
        bin = tf.random_shuffle([0, 1])
        if bin[0] == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
        else:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.divide(tf.subtract(tf.image.resize_images(image, [p.IMAGE_HEIGHT, p.IMAGE_WIDTH]), p.MEAN_IMAGE), p.STD_IMAGE)
    return image

def read_file(filename):
    """
        Args:
            filename: a scalar string tensor.
        Returns:
            data: decoded file
    """

    file_contents = tf.read_file(filename)
    data = tf.decode_raw(file_contents, out_type=tf.float64)
    data = tf.cast(data[10:], tf.float32)
    return data

def create_file_list(path_to_folder):
    """
    Args:
        path_to_images: full path to image folder
    Returns:
        image_list: a tensor of all files in that folder
    """

    file_list = os.listdir(path_to_folder)
    file_list = [path_to_folder + s for s in file_list]
    return file_list


def create_batch(batch_size, train):
    """
    Args:
        batch_size: number of examples in mini-batch
        train: boolean for training or testing mode
    Returns:
        batch:  list of tensors (see SqueezeDet paper for more details) -
                images, 4d tensor sz = [batch_sz, im_h, im_w, im_d]
                masks, whether or not an anchor is assigned to a GT {1,0}, 2d tensor sz = [batch_sz, no_anchors_per_image]
                deltas, deltas between GT assigned to each anchor and the anchors themselves, 3d tensor sz = [batch_sz, no_anchors_per_image,4]
                coords, coords for GT assigned to each anchor, 3d tensor sz = [batch_sz, no_anchors_per_image,4]
                labels, one hot class labels for GT assigned to each anchor, 3d tensor sz = [batch_sz, no_anchors_per_image,no_classes]
    """
    image_list = create_file_list(p.PATH_TO_IMAGES)
    mask_list = create_file_list(p.PATH_TO_MASK)
    delta_list = create_file_list(p.PATH_TO_DELTAS)
    coord_list = create_file_list(p.PATH_TO_COORDS)
    label_list = create_file_list(p.PATH_TO_CLASSES)

    no_samples = len(image_list)

    if train == False:
        mask_list = delta_list = coord_list = label_list = [0]*no_samples

    image_list = tf.convert_to_tensor(image_list, dtype=tf.string)
    mask_list = tf.convert_to_tensor(mask_list, dtype=tf.string)
    delta_list = tf.convert_to_tensor(delta_list, dtype=tf.string)
    coord_list = tf.convert_to_tensor(coord_list, dtype=tf.string)
    label_list = tf.convert_to_tensor(label_list, dtype=tf.string)

    input_queue = tf.train.slice_input_producer([image_list, mask_list, delta_list, coord_list, label_list], shuffle=True)

    images = read_image(input_queue[0])

    masks = read_file(input_queue[1])
    masks = tf.reshape(masks, [p.NR_ANCHORS_PER_IMAGE,1 ])

    deltas = read_file(input_queue[2])
    deltas = tf.transpose(tf.reshape(deltas, [4, p.NR_ANCHORS_PER_IMAGE]))

    coords = read_file(input_queue[3])
    coords = tf.reshape(coords, [p.NR_ANCHORS_PER_IMAGE, 4])

    labels = read_file(input_queue[4])
    labels = tf.reshape(labels, [p.NR_ANCHORS_PER_IMAGE, p.NR_CLASSES])

    batch = tf.train.batch([images, masks, deltas, coords, labels], batch_size=batch_size)

    return batch