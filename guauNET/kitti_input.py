import tensorflow as tf
import os
import numpy as np
import time
import tools as t

from parameters import IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES, ANCHORS, NR_ANCHORS_PER_IMAGE

# TO DO!
# GT_MASK: 1 if anchor has max IOU with GT [batch size, no anchors per image]
#               - compute IOU between each anchor and each GT
#               - for each anchors, find GT with highest IOU
# GT_BOX_DELTA: deltas between anchors and GT with highest IOU [batch size, no anchors per image, 4]
# GT_BOX_INPUT: coordinates of GT bbox assigned to each anchor [batch size, no anchors per image, 4]
# GT_LABELS: labels of GT assigned to each anchor [batch size, no anchors per image, no classes]
# GT_OBJ_PER_IMAGE: number of objects per image [batch size]

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
        bbox: a 1D list of tuples of ground truth bounding boxes (x, y, w, h), list_sz = [total_no_images]
        classes: a 1D list of arrays of ground truth class labels, list_sz = [total_no_images], array_ sz = [number_of_objects_per_image]
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
    return classes,bboxes


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

def compute_deltas(coords):
    coords = np.array(coords)
    anchors = np.array(ANCHORS)
    delta_x = (coords[:,0] - anchors[:,0])/anchors[:,2]
    print(np.shape(delta_x))

def assign_gt_to_anchors(classes, bboxes):
    """
    Args:
        classes:
        bboxes:
    Returns:
         gt_mask: a 2d array with values 1 if the anchor has the highest IOU with an obj, 0 otherwise, sz = [no_images,no_anchors]
         gt_deltas:
         gt_coords:
         gt_labels:
    """
    for image_idx in range(0, len(classes)):
        ious = []
        mask = np.zeros([NR_ANCHORS_PER_IMAGE])
        for obj in bboxes[image_idx]:
            ious_obj = t.compute_iou(np.transpose(ANCHORS), np.transpose(obj))
            ious.append(ious_obj)
        obj_idx_for_anchor = np.argmax(ious, axis=0)
        anchor_idx_for_obj = np.argmax(ious, axis=1)
        mask[anchor_idx_for_obj] = 1
        im_coords = bboxes[image_idx]
        coords = [im_coords[i] for i in obj_idx_for_anchor[:]]
        compute_deltas(coords)


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
       gt_mask, gt_deltas, gt_coords, gt_labels = assign_gt_to_anchors(classes, bboxes)
    else:   # Create fake labels for testing data
        classes = [0]*no_samples
        bboxes = [0]*no_samples
    classes = tf.convert_to_tensor(classes, dtype=tf.int32)
    bboxes = tf.convert_to_tensor(bboxes, dtype=tf.float32)
    input_queue = tf.train.slice_input_producer([image_list, classes, bboxes], shuffle=False)
    images = read_image(input_queue[0])
    batch = tf.train.batch([images, input_queue[1], input_queue[2]], batch_size=batch_size)
    return batch

