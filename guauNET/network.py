import tensorflow as tf
from parameters import NR_CLASSES, NR_ANCHORS_PER_CELL

# variables

def weight_variable(shape, name):
    weights = tf.get_variable(
        name, shape, initializer=tf.contrib.layers.xavier_initializer())
    tf.summary.histogram(name, weights)
    return weights

def gate_weight_variable(shape, name):
    weights = -1*tf.abs(tf.get_variable(
        name, shape, initializer=tf.contrib.layers.xavier_initializer()))
    tf.summary.histogram(name, weights)
    return weights

def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape, name=name)
    bias = tf.Variable(initial)
    tf.summary.histogram(name, bias)
    return bias

# operations

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='Conv')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1], padding='VALID', name='MaxPool')

# blocks

def forget_fire(x_prev, input_depth, s1x1, e1x1, e3x3, name):
    with tf.variable_scope(name):

        with tf.variable_scope('Squeeze'):
            W_s = weight_variable([1, 1, input_depth, s1x1], 'Weights1x1')
            b_s = bias_variable([s1x1], 'Bias1x1')
            h_s = tf.nn.relu(conv2d(x_prev, W_s) + b_s, name='ReLU')

        with tf.variable_scope('Expand'):
            W_e1x1 = weight_variable([1, 1, s1x1, e1x1], 'Weights1x1')
            b_e1x1 = bias_variable([e1x1], 'Bias1x1')
            h_e1x1 = tf.nn.bias_add(conv2d(h_s, W_e1x1), b_e1x1)
            h_e1x1 = tf.nn.relu(h_e1x1)

            W_e3x3 = weight_variable([3, 3, s1x1, e3x3], 'Weights3x3')
            b_e3x3 = bias_variable([e3x3], 'Bias3x3')
            h_e3x3 = tf.nn.bias_add(conv2d(h_s, W_e3x3), b_e3x3)
            h_e3x3 = tf.nn.relu(h_e3x3)

        x_curr = tf.concat([h_e1x1, h_e3x3], 3, name='Concatenate')

        with tf.variable_scope('Forget'):
            h_f_in = tf.concat([x_prev, x_curr], 3, name='Concatenate')
            W_f = gate_weight_variable([3, 3, input_depth + 2 * e3x3, 1], 'Weights1x1')
            b_f = bias_variable([1], 'Bias1x1')
            h_f_out = tf.nn.bias_add(conv2d(h_f_in, W_f), b_f)
            h_f_sig = tf.sigmoid(h_f_out)
            output = x_prev*(1 - h_f_sig) + x_curr*h_f_sig

        tf.summary.histogram('Activation', output)
        return output

def res_asym_fire(x, input_depth, s1x1, e1x1, e3x1, name):
    with tf.variable_scope(name):

        with tf.variable_scope('Squeeze'):
            W_s = weight_variable([1, 1, input_depth, s1x1], 'Weights1x1')
            b_s = bias_variable([s1x1], 'Bias1x1')
            h_s = tf.nn.relu(conv2d(x, W_s) + b_s, name='ReLU')

        with tf.variable_scope('Expand'):
            W_e1x1 = weight_variable([1, 1, s1x1, e1x1], 'Weights1x1')
            b_e1x1 = bias_variable([e1x1], 'Bias1x1')
            h_e1x1 = tf.nn.bias_add(conv2d(h_s, W_e1x1), b_e1x1)
            h_e1x1 = tf.nn.relu(h_e1x1)

            W_e3x1 = weight_variable([3, 1, s1x1, e3x1], 'Weights3x1')
            b_e3x1 = bias_variable([e3x1], 'Bias3x1')
            h_e3x1 = tf.nn.bias_add(conv2d(h_s, W_e3x1), b_e3x1)
            W_e1x3 = weight_variable([1, 3, e3x1, e3x1], 'Weights1x3')
            b_e1x3 = bias_variable([e3x1], 'Bias1x3')
            h_e1x3 = tf.nn.bias_add(conv2d(h_e3x1, W_e1x3), b_e1x3)
            h_e1x3 = tf.nn.relu(h_e1x3)

        output = tf.add(tf.concat([h_e1x1, h_e1x3], 3, name='Concatenate'), x, name='Residual')
        tf.summary.histogram('Activation', output)
        return output

def asym_fire(x, input_depth, s1x1, e1x1, e3x1, name):
    with tf.variable_scope(name):

        with tf.variable_scope('Squeeze'):
            W_s = weight_variable([1, 1, input_depth, s1x1], 'Weights1x1')
            b_s = bias_variable([s1x1], 'Bias1x1')
            h_s = tf.nn.relu(conv2d(x, W_s) + b_s, name='ReLU')

        with tf.variable_scope('Expand'):
            W_e1x1 = weight_variable([1, 1, s1x1, e1x1], 'Weights1x1')
            b_e1x1 = bias_variable([e1x1], 'Bias1x1')
            h_e1x1 = tf.nn.bias_add(conv2d(h_s, W_e1x1), b_e1x1)
            h_e1x1 = tf.nn.relu(h_e1x1)

            W_e3x1 = weight_variable([3, 1, s1x1, e3x1], 'Weights3x1')
            b_e3x1 = bias_variable([e3x1], 'Bias3x1')
            h_e3x1 = tf.nn.bias_add(conv2d(h_s, W_e3x1), b_e3x1)
            W_e1x3 = weight_variable([1, 3, e3x1, e3x1], 'Weights1x3')
            b_e1x3 = bias_variable([e3x1], 'Bias1x3')
            h_e1x3 = tf.nn.bias_add(conv2d(h_e3x1, W_e1x3), b_e1x3)
            h_e1x3 = tf.nn.relu(h_e1x3)

        output = tf.concat([h_e1x1, h_e1x3], 3, name='Concatenate')
        tf.summary.histogram('Activation', output)
        return output

def res_fire(x, input_depth, s1x1, e1x1, e3x3, name):
    with tf.variable_scope(name):

        with tf.variable_scope('Squeeze'):
            W_s = weight_variable([1, 1, input_depth, s1x1], 'Weights1x1')
            b_s = bias_variable([s1x1], 'Bias1x1')
            h_s = tf.nn.relu(conv2d(x, W_s) + b_s, name='ReLU')

        with tf.variable_scope('Expand'):
            W_e1x1 = weight_variable([1, 1, s1x1, e1x1], 'Weights1x1')
            b_e1x1 = bias_variable([e1x1], 'Bias1x1')
            h_e1x1 = tf.nn.bias_add(conv2d(h_s, W_e1x1), b_e1x1)
            h_e1x1 = tf.nn.relu(h_e1x1)

            W_e3x3 = weight_variable([3, 3, s1x1, e3x3], 'Weights3x3')
            b_e3x3 = bias_variable([e3x3], 'Bias3x3')
            h_e3x3 = tf.nn.bias_add(conv2d(h_s, W_e3x3), b_e3x3)
            h_e3x3 = tf.nn.relu(h_e3x3)

        output = tf.add(tf.concat([h_e1x1, h_e3x3], 3, name='Concatenate'), x, 'Residual')
        tf.summary.histogram('Activation', output)
        return output

def fire(x, input_depth, s1x1, e1x1, e3x3, name):
    with tf.variable_scope(name):

        with tf.variable_scope('Squeeze'):
            W_s = weight_variable([1, 1, input_depth, s1x1], 'Weights1x1')
            b_s = bias_variable([s1x1], 'Bias1x1')
            h_s = tf.nn.relu(conv2d(x, W_s) + b_s, name='ReLU')

        with tf.variable_scope('Expand'):
            W_e1x1 = weight_variable([1, 1, s1x1, e1x1], 'Weights1x1')
            b_e1x1 = bias_variable([e1x1], 'Bias1x1')
            h_e1x1 = tf.nn.bias_add(conv2d(h_s, W_e1x1), b_e1x1)
            h_e1x1 = tf.nn.relu(h_e1x1)

            W_e3x3 = weight_variable([3, 3, s1x1, e3x3], 'Weights3x3')
            b_e3x3 = bias_variable([e3x3], 'Bias3x3')
            h_e3x3 = tf.nn.bias_add(conv2d(h_s, W_e3x3), b_e3x3)
            h_e3x3 = tf.nn.relu(h_e3x3)

        output = tf.concat([h_e1x1, h_e3x3], 3, name='Concatenate')
        tf.summary.histogram('Activation', output)
        return output

# architectures

def forget_squeeze_net(x, keep_prop):
    with tf.variable_scope('CNN'):
        with tf.variable_scope('Conv1'):
            W_conv1 = weight_variable([3, 3, 3, 64], 'Weights')
            b_conv1 = bias_variable([64], 'Bias')
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 2, 2, 1], padding='VALID', name='Conv') + b_conv1, name='ReLU')
            h_pool1 = max_pool_3x3(h_conv1)

        h_fire1 = fire(h_pool1, 64, s1x1=16, e1x1=64, e3x3=64, name='Fire1')
        h_fire2 = forget_fire(h_fire1, 128, s1x1=16, e1x1=64, e3x3=64, name='Fire2')
        h_pool2 = max_pool_3x3(h_fire2)

        h_fire3 = fire(h_pool2, 128, s1x1=32,e1x1=128, e3x3=128, name='Fire3')
        h_fire4 = forget_fire(h_fire3, 256, s1x1=32,e1x1=128, e3x3=128, name='Fire4')
        h_pool3 = max_pool_3x3(h_fire4)

        h_fire5 = fire(h_pool3, 256, s1x1=48, e1x1=192, e3x3=192, name='Fire5')
        h_fire6 = forget_fire(h_fire5, 384, s1x1=48, e1x1=192, e3x3=192, name='Fire6')
        h_fire7 = fire(h_fire6, 384, s1x1=64, e1x1=256, e3x3=256, name='Fire7')
        h_fire8 = forget_fire(h_fire7, 512, s1x1=64, e1x1=256, e3x3=256, name='Fire8')
        h_fire9 = fire(h_fire8,  512, s1x1=96, e1x1=384, e3x3=384, name='Fire9')
        h_fire10 = forget_fire(h_fire9, 768, s1x1=96, e1x1=384, e3x3=384, name='Fire10')

        with tf.variable_scope('Dropout'):
            h_drop = tf.nn.dropout(h_fire10, keep_prop, name='Dropout')

        with tf.variable_scope('Conv2'):
            W_conv3 = weight_variable([3, 3, 768, (NR_CLASSES+1+4)*NR_ANCHORS_PER_CELL], 'Weights')
            b_conv3 = bias_variable([(NR_CLASSES+1+4)*NR_ANCHORS_PER_CELL], 'Bias')
            h_conv3 = tf.nn.bias_add(conv2d(h_drop, W_conv3), b_conv3, name='AddBias')

    return h_conv3

def res_asym_squeeze_net(x, keep_prop):
    with tf.variable_scope('CNN'):
        with tf.variable_scope('Conv1'):
            W_conv1 = weight_variable([3, 3, 3, 64], 'Weights')
            b_conv1 = bias_variable([64], 'Bias')
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 2, 2, 1], padding='VALID', name='Conv') + b_conv1, name='ReLU')
            h_pool1 = max_pool_3x3(h_conv1)

        h_fire1 = asym_fire(h_pool1, 64, s1x1=16, e1x1=64, e3x1=64, name='Fire1')
        h_fire2 = res_asym_fire(h_fire1, 128, s1x1=16, e1x1=64, e3x1=64, name='Fire2')
        h_pool2 = max_pool_3x3(h_fire2)

        h_fire3 = asym_fire(h_pool2, 128, s1x1=32,e1x1=128, e3x1=128, name='Fire3')
        h_fire4 = res_asym_fire(h_fire3, 256, s1x1=32,e1x1=128, e3x1=128, name='Fire4')
        h_pool3 = max_pool_3x3(h_fire4)

        h_fire5 = asym_fire(h_pool3, 256, s1x1=48, e1x1=192, e3x1=192, name='Fire5')
        h_fire6 = res_asym_fire(h_fire5, 384, s1x1=48, e1x1=192, e3x1=192, name='Fire6')
        h_fire7 = asym_fire(h_fire6, 384, s1x1=64, e1x1=256, e3x1=256, name='Fire7')
        h_fire8 = res_asym_fire(h_fire7, 512, s1x1=64, e1x1=256, e3x1=256, name='Fire8')
        h_fire9 = asym_fire(h_fire8,  512, s1x1=96, e1x1=384, e3x1=384, name='Fire9')
        h_fire10 = res_asym_fire(h_fire9, 768, s1x1=96, e1x1=384, e3x1=384, name='Fire10')

        with tf.variable_scope('Dropout'):
            h_drop = tf.nn.dropout(h_fire10, keep_prop, name='Dropout')

        with tf.variable_scope('Conv2'):
            W_conv3 = weight_variable([3, 3, 768, (NR_CLASSES+1+4)*NR_ANCHORS_PER_CELL], 'Weights')
            b_conv3 = bias_variable([(NR_CLASSES+1+4)*NR_ANCHORS_PER_CELL], 'Bias')
            h_conv3 = tf.nn.bias_add(conv2d(h_drop, W_conv3), b_conv3, name='AddBias')

    return h_conv3

def res_squeeze_net(x, keep_prop):
    with tf.variable_scope('CNN'):
        with tf.variable_scope('Conv1'):
            W_conv1 = weight_variable([3, 3, 3, 64], 'Weights')
            b_conv1 = bias_variable([64], 'Bias')
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 2, 2, 1], padding='VALID', name='Conv') + b_conv1, name='ReLU')
            h_pool1 = max_pool_3x3(h_conv1)

        h_fire1 = fire(h_pool1, 64, s1x1=16, e1x1=64, e3x3=64, name='Fire1')
        h_fire2 = res_fire(h_fire1, 128, s1x1=16, e1x1=64, e3x3=64, name='Fire2')
        h_pool2 = max_pool_3x3(h_fire2)

        h_fire3 = fire(h_pool2, 128, s1x1=32,e1x1=128, e3x3=128, name='Fire3')
        h_fire4 = res_fire(h_fire3, 256, s1x1=32,e1x1=128, e3x3=128, name='Fire4')
        h_pool3 = max_pool_3x3(h_fire4)

        h_fire5 = fire(h_pool3, 256, s1x1=48, e1x1=192, e3x3=192, name='Fire5')
        h_fire6 = res_fire(h_fire5, 384, s1x1=48, e1x1=192, e3x3=192, name='Fire6')
        h_fire7 = fire(h_fire6, 384, s1x1=64, e1x1=256, e3x3=256, name='Fire7')
        h_fire8 = res_fire(h_fire7, 512, s1x1=64, e1x1=256, e3x3=256, name='Fire8')
        h_fire9 = fire(h_fire8,  512, s1x1=96, e1x1=384, e3x3=384, name='Fire9')
        h_fire10 = res_fire(h_fire9, 768, s1x1=96, e1x1=384, e3x3=384, name='Fire10')

        with tf.variable_scope('Dropout'):
            h_drop = tf.nn.dropout(h_fire10, keep_prop, name='Dropout')

        with tf.variable_scope('Conv2'):
            W_conv3 = weight_variable([3, 3, 768, (NR_CLASSES+1+4)*NR_ANCHORS_PER_CELL], 'Weights')
            b_conv3 = bias_variable([(NR_CLASSES+1+4)*NR_ANCHORS_PER_CELL], 'Bias')
            h_conv3 = tf.nn.bias_add(conv2d(h_drop, W_conv3), b_conv3, name='AddBias')

    return h_conv3

def squeeze_net(x, keep_prop):
    with tf.variable_scope('CNN'):
        with tf.variable_scope('Conv1'):
            W_conv1 = weight_variable([3, 3, 3, 64], 'Weights')
            b_conv1 = bias_variable([64], 'Bias')
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 2, 2, 1], padding='VALID', name='Conv') + b_conv1, name='ReLU')
            h_pool1 = max_pool_3x3(h_conv1)

        h_fire1 = fire(h_pool1, 64, s1x1=16, e1x1=64, e3x3=64, name='Fire1')
        h_fire2 = fire(h_fire1, 128, s1x1=16, e1x1=64, e3x3=64, name='Fire2')
        h_pool2 = max_pool_3x3(h_fire2)

        h_fire3 = fire(h_pool2, 128, s1x1=32,e1x1=128, e3x3=128, name='Fire3')
        h_fire4 = fire(h_fire3, 256, s1x1=32,e1x1=128, e3x3=128, name='Fire4')
        h_pool3 = max_pool_3x3(h_fire4)

        h_fire5 = fire(h_pool3, 256, s1x1=48, e1x1=192, e3x3=192, name='Fire5')
        h_fire6 = fire(h_fire5, 384, s1x1=48, e1x1=192, e3x3=192, name='Fire6')
        h_fire7 = fire(h_fire6, 384, s1x1=64, e1x1=256, e3x3=256, name='Fire7')
        h_fire8 = fire(h_fire7, 512, s1x1=64, e1x1=256, e3x3=256, name='Fire8')
        h_fire9 = fire(h_fire8,  512, s1x1=96, e1x1=384, e3x3=384, name='Fire9')
        h_fire10 = fire(h_fire9, 768, s1x1=96, e1x1=384, e3x3=384, name='Fire10')

        with tf.variable_scope('Dropout'):
            h_drop = tf.nn.dropout(h_fire10, keep_prop, name='Dropout')

        with tf.variable_scope('Conv2'):
            W_conv3 = weight_variable([3, 3, 768, (NR_CLASSES+1+4)*NR_ANCHORS_PER_CELL], 'Weights')
            b_conv3 = bias_variable([(NR_CLASSES+1+4)*NR_ANCHORS_PER_CELL], 'Bias')
            h_conv3 = tf.nn.bias_add(conv2d(h_drop, W_conv3), b_conv3, name='AddBias')

    return h_conv3

