import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

def max_pool_8x8(x):
    return tf.nn.max_pool(x, ksize=[1, 8, 8, 1],
                            strides=[1, 1, 1, 1], padding='VALID')

def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                            strides=[1, 1, 1, 1], padding='VALID')

def meow_net(x):

    # conv1 1x5
    W_conv1 = weight_variable([5, 1, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    # conv2 5x1
    W_conv2 = weight_variable([1, 5, 32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    # max pool 1
    h_pool1 = max_pool_2x2(h_conv2)

    # conv3 1x5
    W_conv3 = weight_variable([5, 1, 32, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

    # conv4 5x1
    W_conv4 = weight_variable([1, 5, 64, 64])
    b_conv4 = bias_variable([64])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    # max pool 2
    h_pool2 = max_pool_2x2(h_conv4)

    # conv5 3x3
    W_conv5 = weight_variable([3, 3, 64, 10])
    b_conv5 = bias_variable([10])
    h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)

    # max pool W_o x H_o
    h_pool3 = max_pool_8x8(h_conv5)
    h_pool3 = tf.squeeze(h_pool3)

    return h_pool3

def deep_meow_net(x):

    # conv1 1x5
    W_conv1 = weight_variable([5, 1, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    # conv2 5x1
    W_conv2 = weight_variable([1, 5, 32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    # max pool 1
    h_pool1 = max_pool_2x2(h_conv2)

    # conv3 1x5
    W_conv3 = weight_variable([5, 1, 32, 32])
    b_conv3 = bias_variable([32])
    h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

    # conv4 5x1
    W_conv4 = weight_variable([1, 5, 32, 32])
    b_conv4 = bias_variable([32])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    # max pool 2
    h_pool2 = max_pool_2x2(h_conv4)

    # conv5 1x5
    W_conv5 = weight_variable([5, 1, 32, 32])
    b_conv5 = bias_variable([32])
    h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)

    # conv6 5x1
    W_conv6 = weight_variable([1, 5, 32, 32])
    b_conv6 = bias_variable([32])
    h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

    # max pool 3
    h_pool3 = max_pool_2x2(h_conv6)

    # conv7 3x3
    W_conv7 = weight_variable([3, 3, 32, 10])
    b_conv7 = bias_variable([10])
    h_conv7 = tf.nn.relu(conv2d(h_pool3, W_conv7) + b_conv7)

    # max pool W_o x H_o
    h_pool4 = max_pool_4x4(h_conv7)
    h_pool4 = tf.squeeze(h_pool4)

    return h_pool4