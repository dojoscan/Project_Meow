# CALCULATION OF THE LOSS FUNCTION

import tensorflow as tf
import parameters as p
import interpretation as interp
import tools as t

def transform_deltas_to_bbox(net_deltas, train):

    "' Transform the deltas given by the network to x,y,w,h format " \
    "Args:" \
    "   net_deltas: a 3d tensor containing the parameterised offsets for each anchor sz=[batch_sz, no_anchors_per_image,4]" \
    "Returns:" \
    "   if Train = true"\
    "   pred_coords: a 3d tensor containing the transformation of the deltas to x, y, w, h, sz =[batch_sz, no_anchors_per_image,4]"\
    "   if Train = false (test)" \
    "   pred_coords: a 3d tensor containing the transformation of the deltas to xmin, ymin, xmax, ymax, sz =[batch_sz, no_anchors_per_image,4]"
    with tf.variable_scope("TransformDeltasToBbox"):
        pred_x = p.ANCHORS[:, 0]+(p.ANCHORS[:, 2]*net_deltas[:, :, 0])
        pred_y = p.ANCHORS[:, 1]+(p.ANCHORS[:, 3]*net_deltas[:, :, 1])
        pred_w = p.ANCHORS[:, 2]*tf.exp(net_deltas[:, :, 2])
        pred_h = p.ANCHORS[:, 3]*tf.exp(net_deltas[:, :, 3])

        xmin, ymin, xmax, ymax = t.bbox_transform([pred_x, pred_y, pred_w, pred_h])

        # check if the calculated values are inside the image limits
        xmin = tf.minimum(tf.maximum(0.0, xmin), p.IMAGE_WIDTH - 1.0, name='CalcXmin')
        ymin = tf.minimum(tf.maximum(0.0, ymin), p.IMAGE_HEIGHT - 1.0, name='CalcYmin')
        xmax = tf.maximum(tf.minimum(p.IMAGE_WIDTH - 1.0, xmax), 0.0, name='CalcXmax')
        ymax = tf.maximum(tf.minimum(p.IMAGE_HEIGHT - 1.0, ymax), 0.0, name='CalcYmax')

        if train:
            pred_coords = t.bbox_transform_inv([xmin, ymin, xmax, ymax])
            pred_coords = tf.transpose(tf.stack(pred_coords, axis=1), perm=[0,2,1], name='BboxCoords')
        else:
            pred_coords = [xmin, ymin, xmax, ymax]
            pred_coords = tf.transpose(tf.stack(pred_coords, axis=1), perm=[0, 2, 1], name='BboxCoords')
    return pred_coords

def bbox_regression(mask, gt_deltas, net_deltas, nr_objects):
    "' Calculate bbox regression" \
    "Returns:" \
    "   loss: the bbox regression calculated (a number)"
    with tf.variable_scope("BboxLoss"):
        #loss = (p.LAMBDA_BBOX/(nr_objects+p.EPSILON))*tf.reduce_sum(tf.square(mask*(net_deltas-gt_deltas)))
        loss = tf.truediv(tf.reduce_sum(p.LAMBDA_BBOX* tf.square(mask * (net_deltas- gt_deltas))),nr_objects,name='bbox_loss')
    return loss


def confidence_score_regression(mask, confidence_scores, gt_confidence_scores, nr_objects):
    "'Calculate the confidence score regression." \
    "Returns:" \
    "   loss: the confidence score regression (a number)"
    with tf.variable_scope("ObjectConfidenceLoss"):
        input_mask = tf.reshape(mask,[p.BATCH_SIZE, p.NR_ANCHORS_PER_IMAGE])
        #loss=tf.reduce_sum(((p.LAMBDA_CONF_POS/nr_objects)*tf.square(input_mask*(confidence_scores-gt_confidence_scores)))-((p.LAMBDA_CONF_NEG/(p.NR_ANCHORS_PER_IMAGE-nr_objects))*(1-input_mask)*tf.square(confidence_scores)))
        loss = tf.reduce_mean(tf.reduce_sum(tf.square((gt_confidence_scores - confidence_scores)) *
                                             (input_mask*p.LAMBDA_CONF_POS/nr_objects+(1-input_mask) * p.LAMBDA_CONF_NEG /
                                             (p.NR_ANCHORS_PER_IMAGE-nr_objects)), reduction_indices=[1]), name="ObjectConfLoss")

    return loss

def classification_regression(mask, gt_labels, class_score, nr_objects):
    "' Calculates the classification regression." \
    "Args:" \
    "   nr_objects: number of objects in the whole batch" \
    "Returns:" \
    "   loss: the classification regression (a number)"
    with tf.variable_scope("ClassConfidenceLoss"):
        #loss = tf.reduce_sum(mask*(gt_labels*tf.log(class_score+p.EPSILON)))/nr_objects
        loss = tf.truediv(tf.reduce_sum((gt_labels*(-tf.log(class_score+p.EPSILON)) +
                                         (1-gt_labels)*(-tf.log(1-class_score+p.EPSILON)))* mask),nr_objects, name='DeltaLoss')
    return loss

def loss_function(mask, gt_deltas, gt_coords,  net_deltas, net_confidence_scores, gt_labels, net_class_score, train):
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
    with tf.variable_scope("Loss"):
        nr_objects = tf.reduce_sum(mask, name="NrObjects")
        bbox_loss = bbox_regression(mask, gt_deltas, net_deltas, nr_objects)
        net_coords = transform_deltas_to_bbox(net_deltas, train)
        gt_confidence_scores = interp.tensor_iou(net_coords, gt_coords)
        confidence_loss = confidence_score_regression(mask, net_confidence_scores, gt_confidence_scores, nr_objects)
        classification_loss = classification_regression(mask, gt_labels, net_class_score, nr_objects)
        total_loss = bbox_loss + confidence_loss + classification_loss
        l2_loss = p.WEIGHT_DECAY_FACTOR * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        total_loss += l2_loss
    return total_loss, bbox_loss, confidence_loss, classification_loss, l2_loss