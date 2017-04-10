# CONTAINS IOU, BOX TRANSFORM, GET LAST CKPT FUNCTIONS

import numpy as np
import parameters as p
import os
import glob

def compute_iou(boxA,boxB):
    ''' Computes the intersection-over-union of an array of boxes and one box
    Args:
        boxA: a 1D or 2D array encoding the top left, top right, width and height, sz = [no_boxes,4]
        boxB: a 1D or 2D array encoding the top left, top right, width and height, sz = [no_boxes,4]
        Note - atleast one of boxA and boxB be a vector of size [1,4]
    Returns:
         iou: arrays of IOUs between a box (or set of boxes) and another box
    '''

    x_min = np.maximum(boxA[0], boxB[0])
    y_min = np.maximum(boxA[1], boxB[1])
    x_max = np.minimum(boxA[0]+boxA[2], boxB[0]+boxB[2])
    y_max = np.minimum(boxA[1]+boxA[3], boxB[1]+boxB[3])
    w = np.maximum(0.0, x_max - x_min)
    h = np.maximum(0.0, y_max - y_min)
    intersection = w * h
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - intersection

    return intersection / (union + p.EPSILON)

def bbox_transform_inv(bbox):
    """
    Convert a bbox of form (x_min, y_min, x_max, y_max) to (x_c, y_c, w, h)
    """
    xmin, ymin, xmax, ymax = bbox
    out_box = [[]]*4
    width       = xmax - xmin + 1.0
    height      = ymax - ymin + 1.0
    out_box[0]  = xmin + 0.5*width
    out_box[1]  = ymin + 0.5*height
    out_box[2]  = width
    out_box[3]  = height
    return out_box

def bbox_transform(bbox):
    """
    Convert a bbox of form (x_c, y_c, w, h) to (x_min, y_min, x_max, y_max). Works for numpy array or list of tensors.
    """
    cx, cy, w, h = bbox
    out_box = [[]]*4
    out_box[0] = cx-w/2
    out_box[1] = cy-h/2
    out_box[2] = cx+w/2
    out_box[3] = cy+h/2

    return out_box

def get_last_ckpt(path_to_dir):
    """
    Find the latest checkpoint in a directory (ckpt name must contain global step)
    """
    newest = max(glob.iglob(path_to_dir + '*.meta'), key=os.path.getctime)
    split_path = newest.split('.')[0]
    init_step = int(split_path.split('-')[-1])
    path_to_last_ckpt = path_to_dir+split_path

    return path_to_last_ckpt, init_step
