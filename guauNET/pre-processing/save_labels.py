# GENERATION OF KITTI LABELS WHICH ARE SAVED TO DISK BEFORE TRAINING

import tensorflow as tf
import os
import numpy as np
import time
import tools as t
import parameters as p

def read_labels(path_to_labels):
    """
    Reads txt file labels in KITTI format
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
    i = 0
    for file in [path_to_labels + s for s in label_list]:
        data = open(file, 'r').read()
        # separate the values for each object in an image
        line = data.split()
        index_images.append(i)
        j = 0
        for obj in range(0, len(line), 15):
            # extract ground truth coordinates [x,y,w,h] and class
            annot = line[obj + 0]
            # convert classes to a number, eg. 'Pedestrian' to 1
            cls = int(p.CLASSES[annot])
            # only consider cars, pedestrians, cyclists
            if cls < 3:
                xmin = float(line[4 + obj])
                ymin = float(line[5 + obj])
                xmax = float(line[6 + obj])
                ymax = float(line[7 + obj])
                x, y, w, h = t.bbox_transform_inv([xmin, ymin, xmax, ymax])
                # if first object in image create new list
                if j == 0:
                    classes.append([cls])
                    bboxes.append([(x, y, w, h)])
                else:
                    ind = index_images.index(i)
                    classes[ind].append(cls)
                    bboxes[ind].append((x, y, w, h))
                j = j + 1
        i = i + 1
    return classes, bboxes

def compute_deltas(coords):
    """
    Converts GT (x_c, y_c, w, h) to deltas (eq. 3 SqueezeDet paper)
    """
    coords = np.array(coords)
    anchors = np.array(p.ANCHORS)
    delta_x = (coords[:, 0] - anchors[:, 0]) / anchors[:, 2]
    delta_y = (coords[:, 1] - anchors[:, 1]) / anchors[:, 3]
    delta_w = np.log((coords[:, 2] + p.EPSILON) / (anchors[:, 2] + p.EPSILON))
    delta_h = np.log((coords[:, 3] + p.EPSILON) / (anchors[:, 3] + p.EPSILON))
    deltas = np.transpose([delta_x, delta_y, delta_w, delta_h])
    return deltas

def create_files(index,path,data):
    """
    Write labels to disk
    """
    filename = ('%06d' % index)+'.txt'
    with open(os.path.join(path, filename), 'wb') as temp_file:
        np.save(temp_file, data)


def assign_gt_to_anchors(classes, bboxes, path_to_data):
    """
    Assigns ground truths to anchors, computes and prints the labels for each anchor
    """
    for image_idx in range(0, len(classes)):
        ious = []
        mask = np.zeros([p.NR_ANCHORS_PER_IMAGE])
        for obj in bboxes[image_idx]:
            ious_obj = t.compute_iou(np.transpose(p.ANCHORS), np.transpose(obj))
            ious.append(ious_obj)
        obj_idx_for_anchor = np.argmax(ious, axis=0)
        anchor_idx_for_obj = np.argmax(ious, axis=1)
        mask[anchor_idx_for_obj] = 1
        im_coords = bboxes[image_idx]
        coords = [im_coords[i] for i in obj_idx_for_anchor[:]]
        deltas = compute_deltas(coords)
        im_label = classes[image_idx]
        pre_label = np.array([im_label[i] for i in obj_idx_for_anchor[:]])
        label = np.zeros((p.NR_ANCHORS_PER_IMAGE, p.NR_CLASSES))
        label[np.arange(p.NR_ANCHORS_PER_IMAGE), pre_label] = 1
        create_files(image_idx, path_to_data +'mask/', mask)
        create_files(image_idx, path_to_data + 'delta/', deltas)
        create_files(image_idx, path_to_data + 'coord/', coords)
        create_files(image_idx, path_to_data + 'class/', label)

def create_label(path_to_data):
    """ Creates and saves binary files of labels (deltas, masks, coordinates, and classes)
    Args:
        path_to_labels: full path to input labels folder
    """

    classes, bboxes = read_labels(path_to_data + 'label/')
    assign_gt_to_anchors(classes, bboxes, path_to_data)

create_label(p.PATH_TO_DATA + 'training/')