import tensorflow as tf
import network_function as nf
import kitti_input as ki
import parameters as p
import interpretation as interp
import loss
import os
import time
import numpy as np
cwd = os.getcwd()

# build input graph
with tf.name_scope('InputPipeline'):
    batch_size = tf.placeholder(dtype=tf.int32, name='BatchSize')
    batch = ki.create_batch(p.PATH_TO_IMAGES, p.PATH_TO_LABELS, batch_size, p.TRAIN)
    x = batch[0]
    y_ = batch[1]

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
class_sc, conf_sc, pred_delta = sess.run([class_scores, confidence_scores, bbox_delta], feed_dict={batch_size: p.BATCH_SIZE})

# move GT interp to input pipeline
# calculate IOU
# assign detections to GT
# calculate loss
# send loss into training graph