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
