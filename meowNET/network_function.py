import tensorflow as tf

KEEP_PROP=0.5

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

def max_pool_2x2_WxH(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 1, 1, 1], padding='VALID')

def normalization(x):
    return tf.nn.local_response_normalization(x)

def conv_pool(x,D1i,D1o,D2o):
    # conv1 1x5
    W_conv1 = weight_variable([5, 1, D1i, D1o])
    b_conv1 = bias_variable([D1o])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    # conv2 5x1
    W_conv2 = weight_variable([1, 5, D1o, D2o])
    b_conv2 = bias_variable([D2o])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    # normalization
    #h_norm1 = normalization(h_conv2)

    # max pool 1
    h_pool1 = max_pool_2x2(h_conv2)

    return h_pool1

################### NETWORK ARCHITECTURES ######################

def meow_net(x):

    h_block1=conv_pool(x,3,32,32)

    h_block2=conv_pool(h_block1,32,64,64)

    # conv5 3x3
    W_conv5 = weight_variable([3, 3, 64, 10])
    b_conv5 = bias_variable([10])
    h_conv5 = tf.nn.relu(conv2d(h_block2, W_conv5) + b_conv5)

    # max pool W_o x H_o
    h_pool3 = max_pool_8x8(h_conv5)
    h_pool3 = tf.squeeze(h_pool3)

    return h_pool3

def deep_meow_net(x):

    h_block1=conv_pool(x,3,32,32)

    h_block2=conv_pool(h_block1,32,32,32)

    h_block3=conv_pool(h_block2,32,32,32)

    # conv7 3x3
    W_conv7 = weight_variable([3, 3, 32, 10])
    b_conv7 = bias_variable([10])
    h_conv7 = tf.nn.relu(conv2d(h_block3, W_conv7) + b_conv7)

    # max pool W_o x H_o
    h_pool4 = max_pool_4x4(h_conv7)
    h_pool4 = tf.squeeze(h_pool4)

    return h_pool4

def deeper_meow_net(x):

    h_block1=conv_pool(x,3,16,16)

    h_block2=conv_pool(h_block1,16,16,16)

    h_block3=conv_pool(h_block2,16,16,16)

    h_block4=conv_pool(h_block3,16,16,16)

    # conv7 3x3
    W_conv9 = weight_variable([3, 3, 16, 10])
    b_conv9 = bias_variable([10])
    h_conv9 = tf.nn.relu(conv2d(h_block4, W_conv9) + b_conv9)

    # max pool W_o x H_o
    h_pool5 = max_pool_2x2_WxH(h_conv9)
    h_pool5 = tf.squeeze(h_pool5)

    return h_pool5

def deep_norm_meow_net(x):

    h_norm1=normalization(x)
    h_block1=conv_pool(h_norm1,3,32,32)

    h_block2=conv_pool(h_block1,32,32,32)

    h_block3=conv_pool(h_block2,32,32,32)

    # conv7 3x3
    W_conv7 = weight_variable([3, 3, 32, 10])
    b_conv7 = bias_variable([10])
    h_conv7 = tf.nn.relu(conv2d(h_block3, W_conv7) + b_conv7)

    #h_norm4=normalization(h_conv7)

    # max pool W_o x H_o
    h_pool4 = max_pool_4x4(h_conv7)
    h_pool4 = tf.squeeze(h_pool4)

    return h_pool4

def deep_dropout_meow_net(x):

    h_block1=conv_pool(x,3,32,32)

    # dropout
    keep_prop = tf.constant(KEEP_PROP, dtype=tf.float32)
    h_drop = tf.nn.dropout(h_block1, keep_prop)

    h_block2=conv_pool(h_drop,32,32,32)

    h_block3=conv_pool(h_block2,32,32,32)

    # conv7 3x3
    W_conv7 = weight_variable([3, 3, 32, 10])
    b_conv7 = bias_variable([10])
    h_conv7 = tf.nn.relu(conv2d(h_block3, W_conv7) + b_conv7)


    # max pool W_o x H_o
    h_pool4 = max_pool_4x4(h_conv7)
    h_pool4 = tf.squeeze(h_pool4)


    return h_pool4

