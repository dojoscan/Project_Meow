# CONTAINS IOU, BOX TRANSFORM FUNCTIONS

import numpy as np
import parameters as p

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
    """ Convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
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
    """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
                                            for numpy array or list of tensors.
    """
    cx, cy, w, h = bbox
    out_box = [[]]*4
    out_box[0] = cx-w/2
    out_box[1] = cy-h/2
    out_box[2] = cx+w/2
    out_box[3] = cy+h/2

    return out_box