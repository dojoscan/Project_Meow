import tensorflow as tf
import network_function as nf
import input
import os
import time
import numpy as np

def conv_pool_tf(x,W1,b1,W2,b2,name):
    with tf.variable_scope(name):
        with tf.variable_scope('5x1'):
            # conv1 1x5
            h_conv1 = tf.nn.relu(nf.conv2d(x, W1) + b1)

        with tf.variable_scope('1x5'):
            # conv2 5x1
            h_conv2 = tf.nn.relu(nf.conv2d(h_conv1, W2) + b2)

        # max pool 1
        h_pool1 = nf.max_pool_2x2(h_conv2)

    return h_pool1

# parameters
LEARNING_RATE = 0.001
NR_ITERATIONS = 1
PRINT_FREQ = 1
BATCH_SIZE = 1
NO_CLASSES = 10
USER = 'LUCIA'
if USER == 'DONAL':
    PATH_TO_IMAGES = "/Users/Donal/Dropbox/CIFAR10/Data/train/images/"
    PATH_TO_LABELS = "/Users/Donal/Dropbox/CIFAR10/Data/train/labels.txt"
    PATH_TO_TEST_IMAGES = "/Users/Donal/Dropbox/CIFAR10/Data/test/images/"
    PATH_TO_LOGS = "/Users/Donal/Desktop/Thesis/Code/Tensorflow_logs/"
    PATH_TO_TEST_OUTPUT = "/Users/Donal/Desktop/predictions.txt"
    PATH_TO_CKPT = "/Users/Donal/Desktop/Thesis/Code/Tensorflow_ckpt/deeper_meow/"
else:
    PATH_TO_IMAGES = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/train1/images/"
    PATH_TO_LABELS = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/train1/labels.txt"
    PATH_TO_TEST_IMAGES = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/test/images/"
    PATH_TO_LOGS = "C:/log_ckpt_thesis/meow_tf/logs/"
    PATH_TO_CKPT="C:/log_ckpt_thesis/meow_tf/ckpt/"

cwd = os.getcwd()

# build input graph
batch_size = tf.placeholder(dtype=tf.int32, name='BatchSize')
batch = input.create_batch(PATH_TO_IMAGES, PATH_TO_LABELS, batch_size, True)
x = batch[0]
gt_labels = tf.one_hot(batch[1], NO_CLASSES, dtype=tf.int32)

# load the old graph
old_saver=tf.train.import_meta_graph('C:/log_ckpt_thesis/meow_tf/ckpt/one_image/deeper_meow.ckpt.meta')
# Access the graph
old_graph = tf.get_default_graph()
# Choose with nodes/operations you want to connect to the new graph and stop gradient
block1_5x1_w = tf.stop_gradient(old_graph.get_tensor_by_name('block1/5x1/Weights:0'))
block1_5x1_b = tf.stop_gradient(old_graph.get_tensor_by_name('block1/5x1/Bias:0'))
block1_1x5_w = tf.stop_gradient(old_graph.get_tensor_by_name('block1/1x5/Weights:0'))
block1_1x5_b = tf.stop_gradient(old_graph.get_tensor_by_name('block1/1x5/Bias:0'))
block2_5x1_w = tf.stop_gradient(old_graph.get_tensor_by_name('block2/5x1/Weights:0'))
block2_5x1_b = tf.stop_gradient(old_graph.get_tensor_by_name('block2/5x1/Bias:0'))
block2_1x5_w = tf.stop_gradient(old_graph.get_tensor_by_name('block2/1x5/Weights:0'))
block2_1x5_b = tf.stop_gradient(old_graph.get_tensor_by_name('block2/1x5/Bias:0'))

# build CNN graph

with tf.variable_scope('Operation_new'):
    h_block1 = conv_pool_tf(x,block1_5x1_w , block1_5x1_b, block1_1x5_w,block1_1x5_b, 'block1')
    h_block2 = conv_pool_tf(h_block1,block2_5x1_w , block2_5x1_b, block2_1x5_w,block2_1x5_b, 'block2')
    h_block3 = nf.conv_pool(h_block2, 32, 32, 32,'block3')
    h_block4 = nf.conv_pool(h_block3, 32, 32, 32,'block4')
    with tf.variable_scope('Conv9'):
        W_conv9 = nf.weight_variable([3, 3, 32, 10], 'Weights')
        b_conv9 = nf.bias_variable([10], 'Bias')
        h_conv9 = tf.nn.relu(nf.conv2d(h_block4, W_conv9) + b_conv9)
    h_pool5 = nf.max_pool_2x2_WxH(h_conv9)
    #h_new= tf.squeeze(h_pool5)
    h_new=h_pool5

# build training graph
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_new, labels=gt_labels))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(h_new, 1), tf.argmax(gt_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

summary_op = tf.summary.scalar("Accuracy", accuracy)

# saver for creating checkpoints
saver = tf.train.Saver()
sess = tf.Session()

# start input queue threads
coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(PATH_TO_LOGS, graph=tf.get_default_graph())

# training
print('Training initiated')
start_time = time.time()
for i in range(NR_ITERATIONS):
    if i % PRINT_FREQ == 0:
        # evaluate forward pass for mini-batch
        train_accuracy, summary, salida = sess.run([accuracy, summary_op, h_block2], feed_dict={batch_size: BATCH_SIZE})
        print(np.mean(salida))
        print(np.std(salida))
        print("step %d, train accuracy = %g, time taken = %g seconds" % (i, train_accuracy, time.time()-start_time))
        # write accuracy to log file
        summary_writer.add_summary(summary, i)
        start_time = time.time()
    # optimise network
    sess.run(train_step, feed_dict={batch_size: BATCH_SIZE})

save_path = saver.save(sess, PATH_TO_CKPT + "deeper_meow_tf")
print("Parameters saved in %s! Final train accuracy = %g" % (PATH_TO_CKPT, sess.run(accuracy, feed_dict={batch_size: BATCH_SIZE})))