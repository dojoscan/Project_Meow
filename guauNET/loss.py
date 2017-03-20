# CALCULATION OF THE LOSS FUNCTION

import tensorflow as tf
import parameters as p
import interpretation as interp
import tools as t

def transform_deltas_to_bbox(net_deltas, train):
    """ Transform the deltas given by the network to x,y,w,h format
    Args:
       net_deltas: a 3d tensor containing the parametrised bbox offsets for each anchor
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
    """ Calculate bbox regression
    Returns:
       loss: the bbox regression calculated (a number)"""

    with tf.variable_scope("BboxLoss"):
        deltas_sum = tf.reduce_sum(tf.square(net_deltas-gt_deltas, name='SquareDiff'), axis=[2], name='SumOverDeltas')
        input_mask = tf.reshape(mask, [-1, p.NR_ANCHORS_PER_IMAGE], name='ReshapeMask')
        masked_deltas = tf.multiply(input_mask, deltas_sum, name='MaskDeltaSum')
        sum_over_anchors = tf.reduce_sum(masked_deltas, axis=[1], name='SumOverAnchor')
        loss = tf.reduce_mean(tf.multiply(tf.truediv(sum_over_anchors, nr_objects, name='NormNoObj'), p.LAMBDA_BBOX, name='MultiplyCoeff'), name='MeanBboxLoss')
    return loss


def confidence_score_regression(mask, confidence_scores, gt_confidence_scores, nr_objects):
    """Calculate the confidence score regression.
    Returns:
       loss: the confidence score regression (a scalar)"""

    with tf.variable_scope("ObjectConfidenceLoss"):
        input_mask = tf.reshape(mask, [-1, p.NR_ANCHORS_PER_IMAGE], name='ReshapeMask')
        mul_mask = tf.multiply(input_mask, tf.square(confidence_scores-gt_confidence_scores, name='SqDiff'), name='Mask')
        obj_norm = tf.truediv(tf.reduce_sum(mul_mask, axis=[1], name='SumOverAnchors'), nr_objects, 'NormNoObj')
        obj_loss = tf.multiply(obj_norm, p.LAMBDA_CONF_POS, name='MultiplyCoeff')
        neg_mask = 1-input_mask
        non_obj_masked = tf.multiply(neg_mask, tf.square(confidence_scores, name='SqNonObjConf'), name='Mask')
        non_obj_sum = tf.reduce_sum(non_obj_masked, axis=[1], name='SumOverAnchors')
        non_obj_norm = tf.truediv(non_obj_sum, p.NR_ANCHORS_PER_IMAGE - nr_objects, name='Norm')
        non_obj_mult = tf.multiply(non_obj_norm, p.LAMBDA_CONF_NEG, name='MultiplyCoeff')
        sum_terms = tf.add(obj_loss, non_obj_mult, name='AddObjNonObj')
        loss = tf.reduce_mean(sum_terms, name="MeanObjectConfLoss")
    return loss

def classification_regression(mask, gt_labels, class_score, nr_objects):
    """ Calculates the classification regression.
    Args:
       nr_objects: number of objects in each image
    Returns:
       loss: the classification regression (a number)"""

    with tf.variable_scope("ClassConfidenceLoss"):
        log_classes = -tf.log(class_score+p.EPSILON, name='NegLogClass')
        labels_by_preds = tf.multiply(gt_labels, log_classes, name='MultipyLabelsPred')
        sum_class = tf.reduce_sum(labels_by_preds, axis=[2], name='SumOverClasses')
        input_mask = tf.reshape(mask, [-1, p.NR_ANCHORS_PER_IMAGE], name='ReshapeMask')
        anchors_sum = tf.reduce_sum(tf.multiply(input_mask, sum_class, name='Mask'), axis=[1], name='SumOverAnchors')
        norm_sum = tf.truediv(anchors_sum, nr_objects, name='NormByNoObj')
        loss = tf.reduce_mean(norm_sum, name='MeanDeltaLoss')
    return loss

def loss_function(mask, gt_deltas, gt_coords,  net_deltas, net_confidence_scores, gt_labels, net_class_score, train):
    """ Calculate the total loss for the network

    Args:
       mask: whether or not an anchor is assigned to a GT {1,0}, 2d tensor sz = [batch_sz, no_anchors_per_image]
       gt_deltas: deltas between coordinates of GT assigned to each anchor and the anchors themselves,
                                                        3d tensor sz = [batch_sz, no_anchors_per_image,4]
       gt_coords: coords of GT assigned to each anchor, 3d tensor sz = [batch_sz, no_anchors_per_image,4]
       net_deltas:  the coord offsets generated by the network for each anchor [batch_sz, no_anchors_per_image,4]
       net_confidence_scores: the Conf(Obj)*IOU generated by the network for each anchor [batch_sz, no_anchors_per_image]
       gt_labels: one hot class labels for GT assigned to each anchor, 3d tensor
                                sz = [batch_sz, no_anchors_per_image,no_classes]
       net_class_score: the softmaxed Conf(Cl|Obj) generated by the network for each anchor
                                        sz =[batch_sz, no_anchors_per_image, no_classes]

    Returns:
       total_loss: a number representing the sum of the four different losses averaged over the batch
                                (L2 [weight decay], bbox coordinate, object confidence and classification confidence)"""

    with tf.variable_scope("Loss"):
        nr_objects = tf.reduce_sum(tf.reshape(mask,[-1, p.NR_ANCHORS_PER_IMAGE]), axis=[1], name="NrObjectsPerImage")
        bbox_loss = bbox_regression(mask, gt_deltas, net_deltas, nr_objects)
        net_coords = transform_deltas_to_bbox(net_deltas, train)
        gt_confidence_scores = interp.tensor_iou(net_coords, gt_coords)
        confidence_loss = confidence_score_regression(mask, net_confidence_scores, gt_confidence_scores, nr_objects)
        classification_loss = classification_regression(mask, gt_labels, net_class_score, nr_objects)
        l2_loss = p.WEIGHT_DECAY_FACTOR * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        total_loss = l2_loss + bbox_loss + confidence_loss + classification_loss

        # summaries for TensorBoard
        tf.summary.scalar('Total_loss', total_loss)
        tf.summary.scalar('Bounding_box_loss', bbox_loss)
        tf.summary.scalar('Object_confidence_loss', confidence_loss)
        tf.summary.scalar('Classification_loss', classification_loss)
        tf.summary.scalar('Weight_decay_loss', l2_loss)

    return total_loss, bbox_loss, confidence_loss, classification_loss, l2_loss