import tensorflow as tf
import parameters as p

def weight_variable(shape,name, freeze):
    if freeze:
        weights=tf.Variable(tf.zeros(shape), trainable=False, name=name)
    else:
        weights = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    return weights

def bias_variable(shape,name, freeze):
    if freeze:
        bias=tf.Variable(tf.zeros(shape), trainable=False, name=name)
    else:
        initial = tf.constant(0.0, shape=shape)
        bias = tf.Variable(initial, name=name, trainable=True)
    return bias

def fire(x, input_depth, s1x1, e1x1, e3x3, name, freeze, var_dict):
    with tf.variable_scope(name):

        with tf.variable_scope('Squeeze'):
            W_s = weight_variable([1, 1, input_depth, s1x1], 'Weights1x1', freeze)
            b_s = bias_variable([s1x1], 'Bias1x1', freeze)
            h_s = tf.nn.relu(tf.nn.conv2d(x, W_s, strides=[1, 1, 1, 1], padding='SAME', name='Conv') + b_s, name='ReLU')

        with tf.variable_scope('Expand'):
            W_e1x1 = weight_variable([1, 1, s1x1, e1x1], 'Weights1x1', freeze)
            b_e1x1 = bias_variable([e1x1], 'Bias1x1', freeze)
            h_e1x1 = tf.nn.bias_add(tf.nn.conv2d(h_s, W_e1x1, strides=[1, 1, 1, 1], padding='SAME', name='Conv1x1'), b_e1x1)
            h_e1x1 = tf.nn.relu(h_e1x1, name='ReLU')

            W_e3x3 = weight_variable([3, 3, s1x1, e3x3], 'Weights3x3', freeze)
            b_e3x3 = bias_variable([e3x3], 'Bias3x3', freeze)
            h_e3x3 = tf.nn.bias_add(tf.nn.conv2d(h_s, W_e3x3, strides=[1, 1, 1, 1], padding='SAME', name='Conv3x3'), b_e3x3)
            h_e3x3 = tf.nn.relu(h_e3x3, name='ReLU')

        output = tf.concat([h_e1x1, h_e3x3], 3, name='Concatenate')
        fire_dict = {v.op.name: v for v in [W_s, b_s, W_e1x1, b_e1x1, W_e3x3, b_e3x3]}
        var_dict.update(fire_dict)
        return output, var_dict

def squeeze(x, keep_prop, freeze_bool):
    with tf.variable_scope('CNN'):
        # in the weight and variables, if the last term is False, it means that they are intitialize randomly, but if it is True, they are restored from a previous run
        with tf.variable_scope('Conv1'):
            W_conv1 = weight_variable([3, 3, 3, 64], 'Weights', freeze_bool)
            b_conv1 = bias_variable([64], 'Bias', freeze_bool)
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 2, 2, 1], padding='VALID', name='Conv') + b_conv1,
                                 name='ReLU')
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3,3, 1], strides=[1, 2, 2, 1], padding='VALID', name='MaxPool1')

        save_var = {v.op.name: v for v in [W_conv1, b_conv1]}

        h_fire1, save_var = fire(h_pool1, 64, s1x1=16, e1x1=64, e3x3=64, name='Fire1', freeze=freeze_bool, var_dict=save_var)
        h_fire2, save_var = fire(h_fire1, 128, s1x1=16, e1x1=64, e3x3=64, name='Fire2', freeze=freeze_bool, var_dict=save_var)
        h_pool2 = tf.nn.max_pool(h_fire2, ksize=[1, 3,3, 1], strides=[1, 2, 2, 1], padding='VALID', name='MaxPool2')

        h_fire3, save_var = fire(h_pool2, 128, s1x1=32, e1x1=128, e3x3=128, name='Fire3', freeze=freeze_bool,var_dict=save_var)
        h_fire4, save_var = fire(h_fire3, 256, s1x1=32, e1x1=128, e3x3=128, name='Fire4', freeze=freeze_bool, var_dict=save_var)
        h_pool3 = tf.nn.max_pool(h_fire4, ksize=[1, 3,3, 1], strides=[1, 2, 2, 1], padding='VALID', name='MaxPool3')

        h_fire5, save_var = fire(h_pool3, 256, s1x1=48, e1x1=192, e3x3=192, name='Fire5', freeze=freeze_bool, var_dict=save_var)
        h_fire6, save_var = fire(h_fire5, 384, s1x1=48, e1x1=192, e3x3=192, name='Fire6', freeze=freeze_bool, var_dict=save_var)
        h_fire7, save_var = fire(h_fire6, 384, s1x1=64, e1x1=256, e3x3=256, name='Fire7', freeze=freeze_bool, var_dict=save_var)
        h_fire8, save_var = fire(h_fire7, 512, s1x1=64, e1x1=256, e3x3=256, name='Fire8', freeze=freeze_bool, var_dict=save_var)

        if p.APPLY_TL:
            h_fire9,_ = fire(h_fire8, 512, s1x1=96, e1x1=384, e3x3=384, name='Fire9', freeze=False)
            h_fire10,_ = fire(h_fire9, 768, s1x1=96, e1x1=384, e3x3=384, name='Fire10', freeze=False)

            with tf.variable_scope('Dropout'):
                h_drop = tf.nn.dropout(h_fire10, keep_prop, name='Dropout')

            depth_fire=768
        else:

            with tf.variable_scope('Dropout'):
                h_drop = tf.nn.dropout(h_fire8, keep_prop, name='Dropout')

            depth_fire = 512

        with tf.variable_scope('Conv2'):
            W_conv3 = weight_variable([3, 3, depth_fire, (p.NR_CLASSES+1+4)*p.NR_ANCHORS_PER_CELL], 'Weights', False)
            b_conv3 = bias_variable([(p.NR_CLASSES+1+4)*p.NR_ANCHORS_PER_CELL], 'Bias', False)
            h_conv3 = tf.nn.bias_add(tf.nn.conv2d(h_drop, W_conv3, strides=[1, 1, 1, 1], padding='SAME', name='Conv'), b_conv3, name='AddBias')
            h_conv3=tf.squeeze(h_conv3)


    return h_conv3, save_var