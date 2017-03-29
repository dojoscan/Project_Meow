# READS BATCHES OF IMAGES AND LABELS (MASKS, COORDS, DELTAS, CLASSES) FROM DISK

import tensorflow as tf
import os
import parameters as p

def read_image(filename, mode):
    """
    Args:
        filename: a scalar string tensor.
    Returns:
        image_tensor: decoded image
    """

    with tf.variable_scope('ReadImage'):
        file_contents = tf.read_file(filename)
        image = tf.image.decode_png(file_contents, channels=3, name='Image')
        with tf.variable_scope('DistortImage'):
            if mode == 'Train':
                bin = tf.random_shuffle([0, 1])
                if bin[0] == 0:
                    image = tf.image.random_brightness(image, max_delta=50. / 255.)
                    image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
                else:
                    image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
                    image = tf.image.random_brightness(image, max_delta=50. / 255.)
        image = tf.image.resize_images(image, [p.SEC_IMAGE_HEIGHT, p.SEC_IMAGE_WIDTH])
        image = tf.subtract(image, tf.reduce_mean(image))
    return image

def read_file(filename):
    """
        Args:
            filename: a scalar string tensor.
        Returns:
            data: decoded file
    """

    with tf.variable_scope('ReadLabel'):
        file_contents = tf.read_file(filename)
        data = tf.decode_raw(file_contents, out_type=tf.float64)
        data = tf.cast(data[10:], tf.float32, name='DecodedLabel')
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


def create_batch(batch_size, mode):
    """
    Args:
        batch_size: number of examples in mini-batch
        mode: 'Train', 'Test' or 'Val'
    Returns:
        batch:  list of tensors (see SqueezeDet paper for more details) -
                images, 4d tensor sz = [batch_sz, im_h, im_w, im_d]
                masks, whether or not an anchor is assigned to a GT {1,0}, 2d tensor sz = [batch_sz, no_anchors_per_image]
                deltas, offsets between GT assigned to each anchor and the anchors themselves, 3d tensor sz = [batch_sz, no_anchors_per_image,4]
                coords, coords for GT assigned to each anchor, 3d tensor sz = [batch_sz, no_anchors_per_image,4]
                labels, one hot class labels for GT assigned to each anchor, 3d tensor sz = [batch_sz, no_anchors_per_image,no_classes]
    """
    with tf.variable_scope('KITTIInputPipeline'):

        if mode == 'Test':
            image_list = create_file_list(p.PATH_TO_TEST_IMAGES)

            no_samples = len(image_list)
            mask_list = delta_list = coord_list = class_list = tf.convert_to_tensor(['0']*no_samples, dtype=tf.string)
            input_queue = tf.train.slice_input_producer([image_list, mask_list, delta_list, coord_list, class_list],
                                                        shuffle=False, name='InputQueue')
            image = read_image(input_queue[0], mode)
            mask = input_queue[1]
            delta = input_queue[2]
            coord = input_queue[3]
            classes = input_queue[4]

        else:
            if mode == 'Train':
                image_list = create_file_list(p.PATH_TO_IMAGES)
                mask_list = create_file_list(p.PATH_TO_MASK)
                delta_list = create_file_list(p.PATH_TO_DELTAS)
                coord_list = create_file_list(p.PATH_TO_COORDS)
                class_list = create_file_list(p.PATH_TO_CLASSES)
            else:   # validation
                image_list = create_file_list(p.PATH_TO_VAL_IMAGES)
                mask_list = create_file_list(p.PATH_TO_VAL_MASK)
                delta_list = create_file_list(p.PATH_TO_VAL_DELTAS)
                coord_list = create_file_list(p.PATH_TO_VAL_COORDS)
                class_list = create_file_list(p.PATH_TO_VAL_CLASSES)

            with tf.variable_scope("ConvertListsToTensor"):
                image_list = tf.convert_to_tensor(image_list, dtype=tf.string)
                mask_list = tf.convert_to_tensor(mask_list, dtype=tf.string)
                delta_list = tf.convert_to_tensor(delta_list, dtype=tf.string)
                coord_list = tf.convert_to_tensor(coord_list, dtype=tf.string)
                class_list = tf.convert_to_tensor(class_list, dtype=tf.string)

            input_queue = tf.train.slice_input_producer([image_list, mask_list, delta_list, coord_list, class_list],
                                                        shuffle=True, name='InputProducer')

            with tf.variable_scope("ReadTensorSlice"):
                image = read_image(input_queue[0], mode)

                mask = read_file(input_queue[1])
                mask = tf.reshape(mask, [p.NR_ANCHORS_PER_IMAGE, 1], name='Masks')

                delta = read_file(input_queue[2])
                delta = tf.transpose(tf.reshape(delta, [4, p.NR_ANCHORS_PER_IMAGE]), name='Deltas')

                coord = read_file(input_queue[3])
                coord = tf.reshape(coord, [p.NR_ANCHORS_PER_IMAGE, 4], name='Coords')

                classes = read_file(input_queue[4])
                classes = tf.reshape(classes, [p.NR_ANCHORS_PER_IMAGE, p.SEC_NR_CLASSES], name='ClassLabels')

        batch = tf.train.batch([image, mask, delta, coord, classes], batch_size=batch_size, name='Batch', num_threads=p.NUM_THREADS)

    return batch