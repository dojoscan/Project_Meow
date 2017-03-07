# TESTING PIPELINE FOR KITTI

import tensorflow as tf
import network as net
import input
import os
import numpy as np
import parameters as p

cwd = os.getcwd()

# build input graph
batch_size = tf.placeholder(dtype=tf.int32)
batch = input.create_batch(p.PATH_TO_IMAGES, p.PATH_TO_LABELS, batch_size, p.TRAIN)
x = batch[0]
y_ = tf.one_hot(batch[1], p.NO_CLASSES, dtype=tf.int32)

# build CNN graph
h_pool3 = net.meow_net(x)

# get prob. dist. over classes
class_prob = tf.nn.softmax(h_pool3)

saver = tf.train.Saver()
sess = tf.Session()

# restore from checkpoint
saver.restore(sess, p.PATH_TO_CKPT)

# start queues
coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)
sess.run(tf.global_variables_initializer())

# run testing
p_c = sess.run(class_prob, feed_dict={batch_size: 5})

output_file = open(p.PATH_TO_TEST_OUTPUT, 'w')
np.savetxt(output_file, p_c)