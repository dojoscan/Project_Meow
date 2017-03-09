# TESTING PIPELINE FOR KITTI

import tensorflow as tf
import network as net
import kitti_input as ki
import parameters as p
import interpretation as interp
import os
import time
import numpy as np
cwd = os.getcwd()

# build input graph
with tf.name_scope('InputPipeline'):
    batch_size = tf.placeholder(dtype=tf.int32, name='BatchSize')
    batch = ki.create_batch(batch_size, p.TRAIN)
    x = batch[0]
    gt_mask = batch[1]
    gt_deltas = batch[2]
    gt_coords = batch[3]
    gt_labels = batch[4]

# build CNN graph
keep_prop = tf.placeholder(dtype = tf.float32, name='KeepProp')
network_output = net.squeeze_net(x, keep_prop)

# build interpretation graph
class_scores, confidence_scores, bbox_delta = interp.interpret(network_output)




#saver = tf.train.Saver()
sess = tf.Session()

# restore from checkpoint
#saver.restore(sess, p.PATH_TO_CKPT)

# start queues
coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)
sess.run(tf.global_variables_initializer())

# run testing
p_c = sess.run(confidence_scores, feed_dict={batch_size: p.BATCH_SIZE, keep_prop: 1})

#output_file = open(p.PATH_TO_TEST_OUTPUT, 'w')
#np.savetxt(output_file, p_c)