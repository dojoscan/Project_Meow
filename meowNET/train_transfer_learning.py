import tensorflow as tf
import network_function as nf
import input
import os
import time
import numpy as np

# parameters
LEARNING_RATE = 0.001
NR_ITERATIONS = 1000
PRINT_FREQ = 50
BATCH_SIZE = 128
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
    PATH_TO_IMAGES = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/train/images/"
    PATH_TO_LABELS = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/train/labels.txt"
    PATH_TO_TEST_IMAGES = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/test/images/"
    PATH_TO_LOGS = "C:/log_ckpt_thesis/MEOW/logs/"
    PATH_TO_CKPT="C:/log_ckpt_thesis/MEOW/ckpt/"

cwd = os.getcwd()

# load the old graph
old_saver=tf.train.import_meta_graph('C:/log_ckpt_thesis/MEOW/ckpt/squeeze_meow/deeper_meow.ckpt.meta')
# Access the graph
old_graph = tf.get_default_graph()
# Choose with nodes/operations you want to connect to the new graph
output_conv = old_graph.get_tensor_by_name('Operation/output:0')
# stop the gradient
output_sg = tf.stop_gradient(output_conv) # It's an identity function

# build input graph
batch_size = tf.placeholder(dtype=tf.int32, name='BatchSize')
batch = input.create_batch(PATH_TO_IMAGES, PATH_TO_LABELS, batch_size, True)
x = batch[0]
gt_labels = tf.one_hot(batch[1], NO_CLASSES, dtype=tf.int32)

# build CNN graph

with tf.variable_scope('Operation_new'):
    W_conv = nf.weight_variable([3, 3, 10, 10])
    b_conv = nf.bias_variable([10])
    h_conv = tf.nn.bias_add(nf.conv2d(output_sg, W_conv) , b_conv, name='output_new')
    h_conv_new = tf.squeeze(h_conv)

# build training graph
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_conv_new, labels=gt_labels))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(h_conv_new, 1), tf.argmax(gt_labels, 1))
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
start_time = time.clock()
for i in range(NR_ITERATIONS):
    if i % PRINT_FREQ == 0:
        # evaluate forward pass for mini-batch
        train_accuracy, summary = sess.run([accuracy, summary_op], feed_dict={batch_size: BATCH_SIZE, 'Placeholder:0': BATCH_SIZE})
        print("step %d, train accuracy = %g, time taken = %g seconds" % (i, train_accuracy, time.clock()-start_time))
        # write accuracy to log file
        summary_writer.add_summary(summary, i)
        start_time = time.clock()
    # optimise network
    sess.run(train_step, feed_dict={batch_size: BATCH_SIZE, 'Placeholder:0': BATCH_SIZE})

save_path = saver.save(sess, PATH_TO_CKPT + "deeper_meow.ckpt")
print("Parameters saved in %s! Final train accuracy = %g" % (PATH_TO_CKPT, sess.run(accuracy, feed_dict={batch_size: BATCH_SIZE, 'Placeholder:0': BATCH_SIZE})))