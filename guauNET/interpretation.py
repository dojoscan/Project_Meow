import tensorflow as tf
import parameters as p

def interpret(network_output):
    '''
    Converts the output tensor of the network to form that can be easily manipulated to calculate the loss
    Args:
        network_output: a 4d tensor outputted from the CNN [batch_sz,height,width,depth]
    Return:
        class_scores: a 3d tensor containing the Pr(Cl|Obj) dist. for each anchor [batch_sz, no_anchors_per_image, no_classes]
        confidence_scores: a 2d tensor containing the Pr(Obj)*IOU for each anchor [batch_sz, no_anchors_per_image]
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
                [p.BATCH_SIZE, p.NR_ANCHORS_PER_CELL*p.OUTPUT_HEIGHT*p.OUTPUT_WIDTH, p.NR_CLASSES],
                name='ClassScores'
            )

        with tf.name_scope('ReformatConfidenceScores'):
            num_confidence_scores = p.NR_ANCHORS_PER_CELL + num_class_probs
            confidence_scores = tf.sigmoid(
                tf.reshape(
                    network_output[ :, :, :, num_class_probs:num_confidence_scores],
                    [p.BATCH_SIZE, p.NR_ANCHORS_PER_CELL*p.OUTPUT_HEIGHT*p.OUTPUT_WIDTH]
                ),
                name='ConfidenceScores'
            )

        with tf.name_scope('ReformatBboxDelta'):
            bbox_delta = tf.reshape(
                network_output[:, :, :, num_confidence_scores:],
                [ p.BATCH_SIZE, p.NR_ANCHORS_PER_CELL*p.OUTPUT_HEIGHT*p.OUTPUT_WIDTH, 4],
                name='bboxDelta'
            )
        return class_scores, confidence_scores, bbox_delta