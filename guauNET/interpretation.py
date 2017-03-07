# CONVERTS OUTPUT OF THE NETWORK TO EASY-TO-WORK-WITH FORMAT

import tensorflow as tf
import parameters as p

def interpret(network_output):
    '''
    Converts the output tensor of the network to form that can be easily manipulated to calculate the loss
    Args:
        network_output: a 4d tensor outputted from the CNN [batch_sz,height,width,depth]
    Return:
        class_scores: a 3d tensor containing the Conf(Cl|Obj) dist. for each anchor [batch_sz, no_anchors_per_image, no_classes]
        confidence_scores: a 2d tensor containing the Conf(Obj)*IOU for each anchor [batch_sz, no_anchors_per_image]
        bbox_deltas: a 3d tensor containing the parameterised offsets for each anchor [batch_sz, no_anchors_per_image,4]
    '''
    with tf.name_scope('Interpretation'):
        with tf.name_scope('ReformatClassScores'):
            num_class_probs = p.NR_ANCHORS_PER_CELL * p.NR_CLASSES
            class_scores = tf.reshape(
                tf.nn.softmax(
                    tf.reshape(
                        network_output[:, :, :, :num_class_probs],
                        [-1, p.NR_CLASSES]
                    )
                ),
                [p.BATCH_SIZE, p.NR_ANCHORS_PER_IMAGE, p.NR_CLASSES],
                name='ClassScores'
            )

        with tf.name_scope('ReformatConfidenceScores'):
            num_confidence_scores = p.NR_ANCHORS_PER_CELL + num_class_probs
            confidence_scores = tf.sigmoid(
                tf.reshape(
                    network_output[:, :, :, num_class_probs:num_confidence_scores],
                    [p.BATCH_SIZE, p.NR_ANCHORS_PER_IMAGE]
                ),
                name='ConfidenceScores'
            )

        with tf.name_scope('ReformatBboxDelta'):
            bbox_delta = tf.reshape(
                network_output[:, :, :, num_confidence_scores:],
                [ p.BATCH_SIZE, p.NR_ANCHORS_PER_IMAGE, 4],
                name='bboxDelta'
            )
        return class_scores, confidence_scores, bbox_delta

def tensor_iou(boxPred, boxGT):
    for obj in boxGT:
        with tf.variable_scope('Intersection'):
            x_min = tf.maximum(boxPred[:, 0], obj[0], name='x_min')
            y_min = tf.maximum(boxPred[:, 1], obj[1], name='y_min')
            x_max = tf.minimum(boxPred[:, 2], obj[2], name='x_max')
            y_max = tf.minimum(boxPred[:, 3], obj[3], name='y_max')
            w = tf.maximum(0.0, x_max - x_min, name='Inter_w')
            h = tf.maximum(0.0, y_max - y_min, name='Inter_h')
            intersection = tf.multiply(w, h, name='Intersection')

        with tf.variable_scope('Union'):
            w1 = tf.subtract(boxPred[:, 2], obj[0], name='w_1')
            h1 = tf.subtract(boxPred[:, 3], obj[1], name='h_1')
            w2 = tf.subtract(boxPred[:, 2], obj[0], name='w_2')
            h2 = tf.subtract(boxPred[:, 3], obj[1], name='h_2')

            union = w1 * h1 + w2 * h2 - intersection

        return intersection / (union + p.EPSILON)
