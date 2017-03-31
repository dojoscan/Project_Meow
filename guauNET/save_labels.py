import tensorflow as tf
import os
import numpy as np
import time
import tools as t
import parameters as p
import sys
sys.path.append("/ProjectMeow/guauNET/")

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
    i = 0
    for file in [path_to_labels + s for s in label_list]:
        data = open(file, 'r').read()
        # separate the values for each image, every image can have more than one object
        line = data.split()
        # an array with the index of each image
        index_images.append(i)
        j = 0
        for obj in range(0, len(line), 15):
            # extract ground truth coordinates [x,y,w,h] and class for each object
            annot = line[obj + 0]
            cls = int(p.CLASSES[annot])  # CLASSES convert classes to a number, eg. 'Pedestrian' to 1
            if cls < 3:
                xmin = float(line[4 + obj])
                ymin = float(line[5 + obj])
                xmax = float(line[6 + obj])
                ymax = float(line[7 + obj])
                x, y, w, h = t.bbox_transform_inv([xmin, ymin, xmax, ymax])
                # create a new classes and bboxes lists, if first object create new list
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
    coords = np.array(coords)
    anchors = np.array(p.ANCHORS)
    delta_x = (coords[:, 0] - anchors[:, 0]) / anchors[:, 2]
    delta_y = (coords[:, 1] - anchors[:, 1]) / anchors[:, 3]
    delta_w = np.log((coords[:, 2] + p.EPSILON) / (anchors[:, 2] + p.EPSILON))
    delta_h = np.log((coords[:, 3] + p.EPSILON) / (anchors[:, 3] + p.EPSILON))
    deltas = np.transpose([delta_x, delta_y, delta_w, delta_h])
    return deltas

def create_files(index,path,data):
    filename = ('%06d'%index)+'.txt'
    with open(os.path.join(path, filename), 'wb') as temp_file:
        np.save(temp_file, data)


def assign_gt_to_anchors(classes, bboxes):
    """
    Assigns ground truths to anchors and computes the labels for each anchor
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
        create_files(image_idx, p.PATH_TO_MASK, mask)
        create_files(image_idx, p.PATH_TO_DELTAS, deltas)
        create_files(image_idx, p.PATH_TO_COORDS, coords)
        create_files(image_idx, p.PATH_TO_CLASSES, label)

def create_label(path_to_labels):
    """ Creates and saves binary files of labels (deltas, masks, coordinates, and classes)
    Args:
        path_to_labels: full path to input labels folder
    """

    classes, bboxes = read_labels(path_to_labels)
    assign_gt_to_anchors(classes, bboxes)

create_label('C:/Master Chalmers/2 year/volvo thesis/code0/training/label/')