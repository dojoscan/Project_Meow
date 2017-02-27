import tensorflow as tf
import numpy as np

KEEP_PROP = 0.5

# variables

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# operations

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

def max_pool_8x8(x):
    return tf.nn.max_pool(x, ksize=[1, 8, 8, 1],
                            strides=[1, 1, 1, 1], padding='VALID')

def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                            strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2_WxH(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 1, 1, 1], padding='VALID')

def normalization(x):
    #return tf.nn.local_response_normalization(x)
    return tf.nn.batch_normalization(x,mean=0,variance=1,offset=0.1,scale=0.99,variance_epsilon=0.00001)

# blocks

def conv_pool(x,D1i,D1o,D2o):
    # conv1 1x5
    W_conv1 = weight_variable([5, 1, D1i, D1o])
    b_conv1 = bias_variable([D1o])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    # conv2 5x1
    W_conv2 = weight_variable([1, 5, D1o, D2o])
    b_conv2 = bias_variable([D2o])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    # max pool 1
    h_pool1 = max_pool_2x2(h_conv2)

    return h_pool1


def fire(x,input_depth,s1x1,e1x1,e3x1):

    #Squeeze layer
    W_s = weight_variable([1, 1, input_depth, s1x1])
    b_s = bias_variable([s1x1])
    h_s = tf.nn.relu(conv2d(x, W_s) + b_s)

    #Expand Layer
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

    return tf.concat([h_e1x1, h_e1x3], 3)

def res_fire(x,input_depth,s1x1,e1x1,e3x1):

    #Squeeze layer
    W_s = weight_variable([1, 1, input_depth, s1x1])
    b_s = bias_variable([s1x1])
    h_s = tf.nn.relu(conv2d(x, W_s) + b_s)

    #Expand Layer
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

    return tf.concat([h_e1x1, h_e1x3], 3)+x

# CNN ARCHITECTURES

def meow_net(x):

    h_block1 = conv_pool(x, 3, 32, 32)
    h_block2 = conv_pool(h_block1, 32, 64, 64)

    # conv5 3x3
    W_conv5 = weight_variable([3, 3, 64, 10])
    b_conv5 = bias_variable([10])
    h_conv5 = tf.nn.relu(conv2d(h_block2, W_conv5) + b_conv5)

    # max pool W_o x H_o
    h_pool3 = max_pool_8x8(h_conv5)
    h_pool3 = tf.squeeze(h_pool3)

    return h_pool3

def deep_meow_net(x):

    h_block1 = conv_pool(x,3,32,32)
    h_block2 = conv_pool(h_block1,32,32,32)
    h_block3 = conv_pool(h_block2,32,32,32)

    # conv7 3x3
    W_conv7 = weight_variable([3, 3, 32, 10])
    b_conv7 = bias_variable([10])
    h_conv7 = tf.nn.relu(conv2d(h_block3, W_conv7) + b_conv7)

    # max pool W_o x H_o
    h_pool4 = max_pool_4x4(h_conv7)
    h_pool4 = tf.squeeze(h_pool4)

    return h_pool4

def deeper_meow_net(x):

    h_block1 = conv_pool(x, 3, 32, 32)
    h_block2 = conv_pool(h_block1, 32, 32, 32)
    h_block3 = conv_pool(h_block2, 32, 32, 32)
    h_block4 = conv_pool(h_block3, 32, 32, 32)

    # conv7 3x3
    W_conv9 = weight_variable([3, 3, 32, 10])
    b_conv9 = bias_variable([10])
    h_conv9 = tf.nn.relu(conv2d(h_block4, W_conv9) + b_conv9)

    # max pool W_o x H_o
    h_pool5 = max_pool_2x2_WxH(h_conv9)
    h_pool5 = tf.squeeze(h_pool5)

    return h_pool5

def deep_dropout_meow_net(x):

    h_block1 = conv_pool(x, 3, 32, 32)

    # dropout
    keep_prop = tf.constant(KEEP_PROP, dtype=tf.float32)
    h_drop = tf.nn.dropout(h_block1, keep_prop)

    h_block2 = conv_pool(h_drop, 32, 32, 32)
    h_block3 = conv_pool(h_block2, 32, 32, 32)

    # conv7 3x3
    W_conv7 = weight_variable([3, 3, 32, 10])
    b_conv7 = bias_variable([10])
    h_conv7 = tf.nn.relu(conv2d(h_block3, W_conv7) + b_conv7)

    # max pool W_o x H_o
    h_pool4 = max_pool_4x4(h_conv7)
    h_pool4 = tf.squeeze(h_pool4)

    return h_pool4

def squeeze_net(x):

    W_conv1 = weight_variable([3, 1, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 2, 2, 1], padding='VALID') + b_conv1)

    W_conv2 = weight_variable([1, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    h_pool1 = max_pool_3x3(h_conv2)

    h_fire1 = res_fire(h_pool1, 64, s1x1=16, e1x1=32, e3x1=32)
    h_fire2 = res_fire(h_fire1, 64, s1x1=16, e1x1=32, e3x1=32)
    h_pool2 = max_pool_3x3(h_fire2)

    h_fire3 = res_fire(h_pool2, 64, s1x1=16, e1x1=32, e3x1=32)
    h_fire4 = fire(h_fire3, 64, s1x1=16, e1x1=32, e3x1=32)
    h_pool3 = max_pool_3x3(h_fire4)

    h_fire5 = res_fire(h_pool3,64,s1x1=32,e1x1=32,e3x1=32)

    W_conv3 = weight_variable([3, 3, 64, 10])
    b_conv3 = bias_variable([10])
    h_conv3 = tf.nn.bias_add(conv2d(h_fire5, W_conv3), b_conv3)
    h_conv3 = tf.squeeze(h_conv3)

    return h_conv3


