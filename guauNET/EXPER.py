import tensorflow as tf
import network_function as nf
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
    gt_coords=batch[3]
    gt_labels=batch[4]

# build CNN graph
network_output = nf.squeeze_net(x)

# build interpretation graph
class_scores, confidence_scores, bbox_delta = interp.interpret(network_output)

sess = tf.Session()

# start input queue threads
with tf.name_scope('Queues'):
    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

sess.run(tf.global_variables_initializer())

# run session
mask, deltas, coords, labels = sess.run([gt_mask,gt_deltas,gt_coords,gt_labels], feed_dict={batch_size: p.BATCH_SIZE})


# move GT interp to input pipeline
# calculate IOU
# assign detections to GT
# calculate loss
# send loss into training graph