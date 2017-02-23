import tensorflow as tf
import numpy as np
from parameters import classes
def bbox_transform_inv(bbox):
    """ Convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
    Args:
        bbox: an array
    Returns:
        outbox: an array
    """
    with tf.variable_scope('bbox_transform_inv') as scope:
        xmin, ymin, xmax, ymax = bbox
        out_box = [[]]*4
        width       = xmax - xmin + 1.0
        height      = ymax - ymin + 1.0
        out_box[0]  = xmin + 0.5*width
        out_box[1]  = ymin + 0.5*height
        out_box[2]  = width
        out_box[3]  = height
    return out_box

def separate_labels(y_):
    labels = {}
    index=0
    for i in range(0,len(y_)):
        line = y_[i].decode()
        bboxes = []
        line=line.split()
        print(line)
        for obj in range(0,len(line),15):
            annot=line[obj+0]
            cls = int(classes[annot])
            xmin = float(line[4+obj])
            ymin = float(line[5+obj])
            xmax = float(line[6+obj])
            ymax = float(line[7+obj])
            x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
            bboxes.append([x, y, w, h, cls])
            print(bboxes)
        labels[index] = bboxes
        index = index+1
    print(labels)
    return labels

def calculate_loss(y_):
    """ Calculate multi-task loss as described in squeezeDet
    Args:
        network_output: a 4d tensor
        y_: a 1d tensor where each element is a string containing annotations for all objects in a single image
    Returns:
        loss: multi-task loss over a mini-Batch
    """
    return separate_labels(y_)