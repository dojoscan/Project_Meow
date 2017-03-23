import tensorflow as tf
import parameters as p

def weight_variable(shape,name, freeze):
    if freeze:
        weights=tf.Variable(tf.zeros(shape), trainable=False)
    else:
        weights = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    return weights

def bias_variable(shape,name, freeze):
    if freeze:
        bias=tf.Variable(tf.zeros(shape), trainable=False)
    else:
        initial = tf.constant(0.0, shape=shape)
        bias = tf.Variable(initial, name=name, trainable=True)
    return bias

def conv_pool(x,D1i,D1o,D2o, name):
    with tf.variable_scope(name):
        with tf.variable_scope('Conv1'):
            # conv1 1x1
            if p.APPLY_TF:
                W_conv1 = tf.Variable(tf.zeros([5, 1, D1i, D1o]), name='Weights', trainable=False)
                salvado=W_conv1
                b_conv1 = tf.Variable(tf.zeros([D1o]), name='Bias', trainable=False)
            else:
                W_conv1 = weight_variable([5, 1, D1i, D1o], 'Weights')
                salvado = W_conv1
                b_conv1 = bias_variable([D1o], 'Bias')

            h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME', name='Conv1') + b_conv1, name='ReLU')

        with tf.variable_scope('Conv2'):
            # conv2 5x1
            if p.APPLY_TF:
                W_conv2 = tf.Variable(tf.zeros([1, 5, D1o, D2o]), name='Weights', trainable=False)
                b_conv2 = tf.Variable(tf.zeros([D2o]), name='Bias', trainable=False)
            else:
                W_conv2 = weight_variable([1, 5, D1o, D2o], 'Weights')
                b_conv2 = bias_variable([D2o], 'Bias')

            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME', name='Conv2') + b_conv2, name='ReLU')

        # max pool 1
        h_pool1 =  tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='MaxPool')
        varis=W_conv1, b_conv1, W_conv2, b_conv2
    return h_pool1, varis, salvado

def simple_net(x):
    with tf.variable_scope('CNN'):
        with tf.variable_scope('first_layer'):
            if p.APPLY_TF:
                # if APPLY_TF=True, initialize weights and bias to zero of the specified layer ans trainable = False
                W1 = tf.Variable(tf.zeros([3,3,3,32]), name='Weights', trainable=False)
                salvado = W1
                b1 = tf.Variable(tf.zeros([32]), name='Bias', trainable=False)
            else:
                W1 = weight_variable([3, 3, 3, 32], 'Weights')
                salvado=W1
                b1 =  bias_variable([32], 'Bias')

            h1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME', name='Conv') + b1, name='ReLU')
            h_pool1 = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='MaxPool')

        with tf.variable_scope('second_layer'):
            W2=weight_variable([3, 3, 32, 10], 'Weights')
            b2 = bias_variable([10], 'Bias')
            h2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W2, strides=[1, 1, 1, 1], padding='SAME', name='Conv') + b2, name='ReLU')

        with tf.variable_scope('last_layer'):
            h_end = tf.nn.max_pool(h2, ksize=[1, 16, 16, 1], strides=[1, 2, 2, 1], padding='VALID', name='MaxPool')
            h_end=tf.squeeze(h_end)

        variables_to_save=[W1, b1]


    return h_end, variables_to_save, salvado

def deeper_net(x):
    with tf.variable_scope('CNN'):

        h_block1, varis_1, salvado1=conv_pool(x,3, 32, 32, 'block1')
        h_block2, varis_2, salvado2=conv_pool(h_block1,32,32,32, 'block2')
        h_block3, varis_3, salvado3 = conv_pool(h_block2, 32, 32, 32, 'block3')

        if p.APPLY_TF:
            with tf.variable_scope('block4'):
                with tf.variable_scope('Conv1'):
                        W_conv1 = weight_variable([5, 1, 32, 32], 'Weights')
                        salvado4 = W_conv1
                        b_conv1 = bias_variable([32], 'Bias')
                        h_conv1 = tf.nn.relu(tf.nn.conv2d(h_block3, W_conv1, strides=[1, 1, 1, 1], padding='SAME', name='Conv1') + b_conv1,
                        name='ReLU')

                with tf.variable_scope('Conv2'):
                    # conv2 5x1
                        W_conv2 = weight_variable([1, 5, 32, 32], 'Weights')
                        b_conv2 = bias_variable([32], 'Bias')
                        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME', name='Conv2') + b_conv2,name='ReLU')

                # max pool 1
                h_block4 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                                         name='MaxPool')

            with tf.variable_scope('fifth_layer'):
                W5 = weight_variable([3, 3, 32, 10], 'Weights')
                b5 = bias_variable([10], 'Bias')
                h5 = tf.nn.relu(tf.nn.conv2d(h_block4, W5, strides=[1, 1, 1, 1], padding='SAME', name='Conv') + b5,
                            name='ReLU')

            with tf.variable_scope('last_layer'):
                h_end = tf.nn.max_pool(h5, ksize=[1, 2,2, 1], strides=[1, 2, 2, 1], padding='VALID', name='MaxPool')
                h_end = tf.squeeze(h_end)

        else:
            salvado4=tf.constant(0.0)
            with tf.variable_scope('fourth_layer'):
                W4 = weight_variable([3, 3, 32, 10], 'Weights')
                b4 = bias_variable([10], 'Bias')
                h4 = tf.nn.relu(tf.nn.conv2d(h_block3, W4, strides=[1, 1, 1, 1], padding='SAME', name='Conv') + b4,
                            name='ReLU')

            with tf.variable_scope('last_layer'):
                h_end = tf.nn.max_pool(h4, ksize=[1, 4,4, 1], strides=[1, 2, 2, 1], padding='VALID', name='MaxPool')
                h_end = tf.squeeze(h_end)

        variables_to_save = varis_1 + varis_2 + varis_3

    return h_end, variables_to_save, salvado1, salvado2, salvado3, salvado4

def fire(x, input_depth, s1x1, e1x1, e3x3, name, freeze):
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
        temp_dict = {v.op.name: v for v in [W_s, b_s, W_e1x1, b_e1x1, W_e3x3, b_e3x3]}
        return output, temp_dict

def squeeze(x, keep_prop):
    with tf.variable_scope('CNN'):
        # in the weight and variables, if the last term is False, it means that they are intitialize randomly, but if it is True, they are restored from a previous run
        with tf.variable_scope('Conv1'):
            W_conv1 = weight_variable([3, 3, 3, 64], 'Weights', False)
            b_conv1 = bias_variable([64], 'Bias', False)
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 2, 2, 1], padding='VALID', name='Conv') + b_conv1,
                                 name='ReLU')
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3,3, 1], strides=[1, 2, 2, 1], padding='VALID', name='MaxPool')

        h_fire1, temp1 = fire(h_pool1, 64, s1x1=16, e1x1=64, e3x3=64, name='Fire1', freeze=False)
        h_fire2, temp2 = fire(h_fire1, 128, s1x1=16, e1x1=64, e3x3=64, name='Fire2', freeze=False)
        h_pool2 = tf.nn.max_pool(h_fire2, ksize=[1, 3,3, 1], strides=[1, 2, 2, 1], padding='VALID', name='MaxPool2')

        h_fire3, temp3 = fire(h_pool2, 128, s1x1=32, e1x1=128, e3x3=128, name='Fire3', freeze=False)
        h_fire4, temp4 = fire(h_fire3, 256, s1x1=32, e1x1=128, e3x3=128, name='Fire4', freeze=False)
        h_pool3 = tf.nn.max_pool(h_fire4, ksize=[1, 3,3, 1], strides=[1, 2, 2, 1], padding='VALID', name='MaxPool3')

        h_fire5, temp5 = fire(h_pool3, 256, s1x1=48, e1x1=192, e3x3=192, name='Fire5', freeze=False)
        h_fire6, temp6 = fire(h_fire5, 384, s1x1=48, e1x1=192, e3x3=192, name='Fire6', freeze=False)
        h_fire7, temp7 = fire(h_fire6, 384, s1x1=64, e1x1=256, e3x3=256, name='Fire7', freeze=False)
        h_fire8, temp8 = fire(h_fire7, 512, s1x1=64, e1x1=256, e3x3=256, name='Fire8', freeze=False)

        if p.APPLY_TF:
            h_fire9 = fire(h_fire8, 512, s1x1=96, e1x1=384, e3x3=384, name='Fire9', freeze=False)
            h_fire10 = fire(h_fire9, 768, s1x1=96, e1x1=384, e3x3=384, name='Fire10', freeze=False)

            with tf.variable_scope('Dropout'):
                h_drop = tf.nn.dropout(h_fire10, keep_prop, name='Dropout')

            depth_fire=768
        else:

            with tf.variable_scope('Dropout'):
                h_drop = tf.nn.dropout(h_fire8, keep_prop, name='Dropout')

            depth_fire = 512

        with tf.variable_scope('Conv2'):
            W_conv3 = weight_variable([3, 3, depth_fire, p.NO_CLASSES], 'Weights', False)
            b_conv3 = bias_variable([p.NO_CLASSES], 'Bias', False)
            h_conv3 = tf.nn.bias_add(tf.nn.conv2d(h_drop, W_conv3, strides=[1, 1, 1, 1], padding='SAME', name='Conv'), b_conv3, name='AddBias')
            h_conv3=tf.squeeze(h_conv3)


        variables_to_save={v.op.name: v for v in [W_conv1, b_conv1]}
        variables_to_save.update(temp1)
    return h_conv3, variables_to_save