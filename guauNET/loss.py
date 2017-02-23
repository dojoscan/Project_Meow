import tensorflow as tf

def calculate_loss(network_output,y_):
    with tf.name_scope('Loss'):

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network_output, labels=y_),
                                       name='CrossEntropy')
        correct_prediction = tf.equal(tf.argmax(network_output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Accuracy')
        tf.summary.scalar("Training Accuracy", accuracy)
        tf.summary.scalar("Cross Entropy", cross_entropy)

        return cross_entropy, accuracy
