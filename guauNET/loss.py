# CALCULATION OF THE LOSS FUNCTION

import tensorflow as tf
import parameters as p
import interpretation as interp
import tools as t

def bbox_regression(mask, gt_deltas, net_deltas, nr_objects):
    "' Calculate bbox regression" \
    "Args: " \
    "   mask: whether or not an anchor is assigned to a GT {1,0}, 2d tensor sz = [batch_sz, no_anchors_per_image" \
    "   gt_deltas: deltas between GT assigned to each anchor and the anchors themselves, 3d tensor sz = [batch_sz, no_anchors_per_image,4]" \
    "   net_deltas: a  3d tensor containing the parameterised offsets for each anchor [batch_sz, no_anchors_per_image,4]" \
    "   nr_objects : number of objects in the whole batch" \
    "Returns:" \
    "   loss: the bbox regression calculated (a number)"
    loss=(p.LAMBDA_BBOX/(nr_objects+p.EPSILON))*tf.reduce_sum(tf.square(mask*(tf.cast(net_deltas,tf.float64)-gt_deltas)))
    return loss

def transform_deltas_to_bbox(net_deltas):

    "' Transform the deltas given by the network to x,y,w,h format " \
    "Args:" \
    "   net_deltas: a 3d tensor containing the parameterised offsets for each anchor [batch_sz, no_anchors_per_image,4]" \
    "Returns:" \
    "   pred_coords: a 3d tensor containing the transformation of the deltas to x, y, w, h, sz=[batch_sz, no_anchors_per_image,4]"

    pred_x = p.ANCHORS[:, 0]+(p.ANCHORS[:, 2]*net_deltas[:, :, 0])
    pred_y = p.ANCHORS[:, 1]+(p.ANCHORS[:, 3]*net_deltas[:, :, 1])
    pred_w = p.ANCHORS[:, 2]*tf.exp(net_deltas[:, :, 2])
    pred_h = p.ANCHORS[:, 3]*tf.exp(net_deltas[:, :, 3])

    xmin, ymin, xmax, ymax = t.bbox_transform([pred_x, pred_y, pred_w, pred_h])

    # check if the calculated values are inside the image limits
    xmin = tf.minimum(tf.maximum(0.0, xmin), p.IMAGE_WIDTH - 1.0)
    ymin = tf.minimum(tf.maximum(0.0, ymin), p.IMAGE_HEIGHT - 1.0)
    xmax = tf.maximum(tf.minimum(p.IMAGE_WIDTH - 1.0, xmax), 0.0)
    ymax = tf.maximum(tf.minimum(p.IMAGE_HEIGHT - 1.0, ymax), 0.0)

    pred_coords = t.bbox_transform_inv([xmin, ymin, xmax, ymax])
    pred_coords = tf.transpose(tf.stack(pred_coords,axis=1), perm=[0,2,1])
    return pred_coords


def confidence_score_regression(mask, confidence_scores, gt_confidence_scores, nr_objects):
    "'Calculate the confidence score regression." \
    "Args:" \
    "   mask: whether or not an anchor is assigned to a GT {1,0}, 2d tensor sz = [batch_sz, no_anchors_per_image" \
    "   confidence_scores: a 2d tensor containing the Conf(Obj)*IOU for each anchor [batch_sz, no_anchors_per_image]" \
    "   gt_confidence_scores: a 2d tensor containing the IOU of the predicted bbox with the ground truth bbox, sz=[batch_sz, no_anchors_per_image]" \
    "   nr_objects: number of objects in the whole batch" \
    "Returns:" \
    "   loss: the confidence score regression (a number)"
    input_mask=tf.reshape(mask,[p.BATCH_SIZE, p.NR_ANCHORS_PER_IMAGE])
    #loss=tf.reduce_sum(((p.LAMBDA_CONF_POS/nr_objects)*tf.square(input_mask*(confidence_scores-gt_confidence_scores)))-((p.LAMBDA_CONF_NEG/(p.NR_ANCHORS_PER_IMAGE-nr_objects))*(1-input_mask)*tf.square(confidence_scores)))
    loss=tf.reduce_mean(tf.reduce_sum(tf.square((gt_confidence_scores - confidence_scores))* (input_mask*p.LAMBDA_CONF_POS/nr_objects+(1-input_mask)*p.LAMBDA_CONF_NEG/(p.NR_ANCHORS_PER_IMAGE-nr_objects)),reduction_indices=[1]))
    return loss

def classification_regression(mask, gt_labels, class_score, nr_objects):
    "' Calculates the classification regression." \
    "Args:" \
    "   mask: whether or not an anchor is assigned to a GT {1,0}, 2d tensor sz = [batch_sz, no_anchors_per_image]" \
    "   gt_labels: one hot class labels for GT assigned to each anchor, 3d tensor sz = [batch_sz, no_anchors_per_image,no_classes]" \
    "   class_score: a 3d tensor containing the Conf(Cl|Obj) dist. for each anchor [batch_sz, no_anchors_per_image, no_classes]" \
    "   nr_objects: number of objects in the whole batch" \
    "Returns:" \
    "   loss: the classification regression (a number)"
    #loss = tf.reduce_sum(mask*(gt_labels*tf.log(class_score+p.EPSILON)))/nr_objects
    loss=tf.truediv(tf.reduce_sum((gt_labels*(-tf.log(class_score+p.EPSILON))+ (1-gt_labels)*(-tf.log(1-class_score+p.EPSILON)))* mask),nr_objects)
    return loss

def loss_function(mask, gt_deltas, gt_coords,  net_deltas, net_confidence_scores, gt_labels, net_class_score):
    "' Calculate the loss for the network." \
    "Args:" \
    "   mask: whether or not an anchor is assigned to a GT {1,0}, 2d tensor sz = [batch_sz, no_anchors_per_image]" \
    "   gt_deltas: deltas between GT assigned to each anchor and the anchors themselves, 3d tensor sz = [batch_sz, no_anchors_per_image,4]" \
    "   gt_coords: coords for GT assigned to each anchor, 3d tensor sz = [batch_sz, no_anchors_per_image,4]" \
    "   net_deltas:  a 3d tensor containing the parameterised offsets for each anchor [batch_sz, no_anchors_per_image,4]" \
    "   net_confidence_scores: a 2d tensor containing the Conf(Obj)*IOU for each anchor [batch_sz, no_anchors_per_image]" \
    "   gt_labels: one hot class labels for GT assigned to each anchor, 3d tensor sz = [batch_sz, no_anchors_per_image,no_classes]" \
    "   net_class_score:  a 3d tensor containing the Conf(Cl|Obj) dist. for each anchor [batch_sz, no_anchors_per_image, no_classes]" \
    "Returns:" \
    "   total_loss: a number representing the sum of the three different losses (bbox, confidence and classification)"
    nr_objects = tf.reduce_sum(mask)
    bbox_loss=bbox_regression(mask,gt_deltas, net_deltas, nr_objects)
    net_coords=transform_deltas_to_bbox(net_deltas)
    # calculate iou between the predicted coordinates and the ground truth coords
    gt_confidence_scores = interp.tensor_iou(tf.cast(net_coords,tf.float64), gt_coords)
    confidence_loss = confidence_score_regression(mask, tf.cast(net_confidence_scores, tf.float64), gt_confidence_scores, nr_objects)
    classification_loss = classification_regression(mask, gt_labels, tf.cast(net_class_score, tf.float64), nr_objects)
    total_loss = bbox_loss + confidence_loss + classification_loss
    return total_loss, bbox_loss, confidence_loss, classification_loss