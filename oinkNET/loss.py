# CALCULATION OF THE LOSS FUNCTION

import tensorflow as tf
import parameters as p
import interpretation as interp
import tools as t

def transform_deltas_to_bbox(net_deltas, train):
    """ Transform the deltas given by the network to x,y,w,h format
    Args:
       net_deltas: a 3d tensor containing the network-predicted parametrised bbox offsets for each anchor
                                                sz=[batch_sz, no_anchors_per_image,4]
    Returns:
       if Train = True
       pred_coords: a 3d tensor containing the transformation of the deltas to x, y, w, h representation,
                                                    sz =[batch_sz, no_anchors_per_image,4]
       if Train = False (test)
       pred_coords: a 3d tensor containing the transformation of the deltas to xmin, ymin, xmax, ymax representation,
                                                            sz =[batch_sz, no_anchors_per_image,4]"""

    with tf.variable_scope("TransformDeltasToBbox"):
        pred_x = p.ANCHORS[:, 0]+(p.ANCHORS[:, 2]*net_deltas[:, :, 0])
        pred_y = p.ANCHORS[:, 1]+(p.ANCHORS[:, 3]*net_deltas[:, :, 1])
        pred_w = p.ANCHORS[:, 2]*tf.exp(net_deltas[:, :, 2])
        pred_h = p.ANCHORS[:, 3]*tf.exp(net_deltas[:, :, 3])

        xmin, ymin, xmax, ymax = t.bbox_transform([pred_x, pred_y, pred_w, pred_h])

        # check if the calculated values are inside the image limits
        with tf.variable_scope("CheckImageBoundary"):
            xmin = tf.minimum(tf.maximum(0.0, xmin), p.SEC_IMAGE_WIDTH - 1.0, name='CalcXmin')
            ymin = tf.minimum(tf.maximum(0.0, ymin), p.SEC_IMAGE_HEIGHT - 1.0, name='CalcYmin')
            xmax = tf.maximum(tf.minimum(p.SEC_IMAGE_WIDTH - 1.0, xmax), 0.0, name='CalcXmax')
            ymax = tf.maximum(tf.minimum(p.SEC_IMAGE_HEIGHT - 1.0, ymax), 0.0, name='CalcYmax')

        if train:
            pred_coords = t.bbox_transform_inv([xmin, ymin, xmax, ymax])
            pred_coords = tf.transpose(tf.stack(pred_coords, axis=1), perm=[0, 2, 1], name='BboxCoords')
        else:
            pred_coords = [xmin, ymin, xmax, ymax]
            pred_coords = tf.transpose(tf.stack(pred_coords, axis=1), perm=[0, 2, 1], name='BboxCoords')
    return pred_coords

def bbox_regression(gt_mask, gt_deltas, net_deltas, nr_objects):
    """ Calculate bbox regression
    Returns:
       loss: the bbox regression calculated (a scalar)"""

    with tf.variable_scope("BboxLoss"):
        bbox_loss = tf.truediv(
            tf.reduce_sum(
                p.LAMBDA_BBOX * tf.square(
                    gt_mask * (net_deltas - gt_deltas))),
            nr_objects,
            name='BboxLoss'
        )
    return bbox_loss


def confidence_score_regression(gt_mask, gt_confidence_scores, net_conf_scores, nr_objects):
    """Calculate the confidence score regression.
    Returns:
       loss: the confidence score regression (a scalar)"""

    with tf.variable_scope("ObjectConfidenceLoss"):
        input_mask = tf.reshape(gt_mask, [p.BATCH_SIZE, p.NR_ANCHORS_PER_IMAGE])
        conf_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.square((gt_confidence_scores - net_conf_scores))
                * (input_mask * p.LAMBDA_CONF_POS / nr_objects
                   + (1 - input_mask) * p.LAMBDA_CONF_NEG / (p.NR_ANCHORS_PER_IMAGE - nr_objects)),
                reduction_indices=[1]
            ),
            name='ConfidenceLoss'
        )
    return conf_loss

def classification_regression(gt_mask, gt_class, net_class_scores, nr_objects):
    """ Calculates the classification regression.
    Args:
       nr_objects: number of objects in each image
    Returns:
       loss: the classification regression (a number)"""

    with tf.variable_scope("ClassLoss"):
        class_loss = tf.truediv(
            tf.reduce_sum(
                (gt_class * (-tf.log(net_class_scores + p.EPSILON))
                 + (1 - gt_class) * (-tf.log(1 - net_class_scores + p.EPSILON)))
                * gt_mask),
            nr_objects,
            name='ClassLoss'
        )
    return class_loss


def loss_function(gt_mask, gt_deltas, gt_coords, gt_class, net_deltas, net_confidence_scores, net_class_score, train):
    """ Calculate the total loss for the network

    Args:
       gt_mask: whether or not an anchor is assigned to a GT {1,0}, 2d tensor sz = [batch_sz, no_anchors_per_image]
       gt_deltas: deltas between coordinates of GT assigned to each anchor and the anchors themselves,
                                                        3d tensor sz = [batch_sz, no_anchors_per_image,4]
       gt_coords: coords of GT assigned to each anchor, 3d tensor sz = [batch_sz, no_anchors_per_image,4]
       gt_class: one hot class labels for GT assigned to each anchor, 3d tensor
                                sz = [batch_sz, no_anchors_per_image,no_classes]
       net_deltas:  the coord offsets generated by the network for each anchor [batch_sz, no_anchors_per_image,4]
       net_confidence_scores: the Conf(Obj)*IOU generated by the network for each anchor [batch_sz, no_anchors_per_image]
       net_class_score: the softmaxed Conf(Cl|Obj) generated by the network for each anchor
                                        sz =[batch_sz, no_anchors_per_image, no_classes]

    Returns:
       total_loss: a number representing the sum of the four different losses averaged over the batch
                                (L2 [weight decay], bbox coordinate, object confidence and classification confidence)"""

    with tf.variable_scope("Loss"):
        nr_objects = tf.reduce_sum(tf.reshape(gt_mask, [p.BATCH_SIZE, p.NR_ANCHORS_PER_IMAGE]), name="NrObjectsPerImage")
        bbox_loss = bbox_regression(gt_mask, gt_deltas, net_deltas, nr_objects)
        net_coords = transform_deltas_to_bbox(net_deltas, train)
        gt_confidence_scores = interp.tensor_iou(net_coords, gt_coords)
        conf_loss = confidence_score_regression(gt_mask, gt_confidence_scores, net_confidence_scores, nr_objects)
        class_loss = classification_regression(gt_mask, gt_class, net_class_score, nr_objects)
        l2_loss = p.WEIGHT_DECAY_FACTOR * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                                    if 'Bias' not in v.name])
        total_loss = bbox_loss + conf_loss + class_loss

    return total_loss, bbox_loss, conf_loss, class_loss, l2_loss

def add_loss_summaries(set, total_loss, bbox_loss, conf_loss, class_loss, l2_loss):

    tot_loss = tf.summary.scalar(set+'Total_loss', total_loss)
    b_loss = tf.summary.scalar(set+'Bounding_box_loss', bbox_loss)
    co_loss = tf.summary.scalar(set+'Object_confidence_loss', conf_loss)
    cl_loss = tf.summary.scalar(set+'Classification_loss', class_loss)
    l_loss = tf.summary.scalar(set+'Weight_decay_loss', l2_loss)

    return tf.summary.merge([tot_loss, b_loss, co_loss, cl_loss, l_loss], name=set+'loss_summary')
