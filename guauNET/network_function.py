import tensorflow as tf
from parameters import NR_CLASSES, NR_ANCHORS_PER_CELL

KEEP_PROP = 0.5

# variables

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, name='Weights')
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, name='Biases')
    return tf.Variable(initial)

# operations
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='Conv')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1], padding='VALID', name='MaxPool')

def squeeze_net(x):

    W_conv1 = weight_variable([3, 1, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 2, 2, 1], padding='VALID', name='Conv') + b_conv1, name='ReLU')

    W_conv2 = weight_variable([1, 3, 32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2, name='ReLU')
    h_pool1 = max_pool_3x3(h_conv2)

    h_fire1 = fire(h_pool1, 32, s1x1=16, e1x1=32, e3x1=32, name='Fire1')
    h_fire2 = fire(h_fire1, 64, s1x1=16, e1x1=32, e3x1=32, name='Fire2')
    h_pool2 = max_pool_3x3(h_fire2)

    h_fire3 = fire(h_pool2, 64, s1x1=16, e1x1=64, e3x1=64, name='Fire3')
    h_fire4 = fire(h_fire3, 128, s1x1=16, e1x1=64, e3x1=64, name='Fire4')
    h_pool3 = max_pool_3x3(h_fire4)

    h_fire5 = fire(h_pool3, 128, s1x1=32, e1x1=128, e3x1=128, name='Fire5')
    h_fire6 = fire(h_fire5, 256, s1x1=48, e1x1=192, e3x1=192, name='Fire6')
    h_fire7 = fire(h_fire6, 384, s1x1=64, e1x1=256, e3x1=256, name='Fire7')
    h_fire8 = fire(h_fire7, 512, s1x1=64, e1x1=256, e3x1=256, name='Fire8')
    h_fire9 = fire(h_fire8, 512, s1x1=96, e1x1=384, e3x1=384, name='Fire9')
    h_fire10 = fire(h_fire9, 768, s1x1=96, e1x1=384, e3x1=384, name='Fire10')
    # dropout
    keep_prop = tf.constant(KEEP_PROP, dtype=tf.float32)
    h_drop = tf.nn.dropout(h_fire10, keep_prop, name='Dropout')

    W_conv3 = weight_variable([3, 3, 768, (NR_CLASSES+1+4)*NR_ANCHORS_PER_CELL])
    b_conv3 = bias_variable([(NR_CLASSES+1+4)*NR_ANCHORS_PER_CELL])
    h_conv3 = tf.nn.bias_add(conv2d(h_drop, W_conv3), b_conv3, name='AddBias')
    h_conv3 = tf.squeeze(h_conv3, name='Squeeze')
    return h_conv3

def fire(x, input_depth, s1x1, e1x1, e3x1, name):
    with tf.variable_scope(name):
        #Squeeze layer
        with tf.variable_scope('Squeeze'):
            W_s = weight_variable([1, 1, input_depth, s1x1])
            b_s = bias_variable([s1x1])
            h_s = tf.nn.relu(conv2d(x, W_s) + b_s, name='ReLU')

        #Expand Layer
        with tf.variable_scope('Expand'):
            W_e1x1 = weight_variable([1, 1, s1x1, e1x1])
            b_e1x1 = bias_variable([e1x1])
            h_e1x1 = tf.nn.bias_add(conv2d(h_s, W_e1x1), b_e1x1)
            h_e1x1 = tf.nn.relu(h_e1x1)

            W_e3x1 = weight_variable([3, 1, s1x1, e3x1])
            b_e3x1 = bias_variable([e3x1])
            h_e3x1 = tf.nn.bias_add(conv2d(h_s, W_e3x1), b_e3x1)
            W_e1x3 = weight_variable([1, 3, e3x1, e3x1])
            b_e1x3 = bias_variable([e3x1])
            h_e1x3 = tf.nn.bias_add(conv2d(h_e3x1, W_e1x3), b_e1x3)
            h_e1x3 = tf.nn.relu(h_e1x3)

        output = tf.concat([h_e1x1, h_e1x3], 3, name='Concatenate')
        tf.summary.histogram(name, output)
        return output
