import tensorflow as tf
import parameters as p

# define as parameters

def interpret(network_output):
    '''

    :param network_output:
    :return:    '''
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