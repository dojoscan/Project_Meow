import tensorflow as tf
import network as net
import kitti_input as ki
import parameters as p
import interpretation as interp
import loss as l
import os
import time
import numpy as np

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
network_output = net.squeeze_net(x)

# build interpretation graph
class_scores, confidence_scores, bbox_delta = interp.interpret(network_output)

total_loss, bbox_loss, confidence_loss, classification_loss = l.loss_function(gt_mask, gt_deltas, gt_coords,  bbox_delta, confidence_scores, gt_labels, class_scores)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(p.LEARNING_RATE, global_step,
                                           p.DECAY_STEP, p.DECAY_FACTOR, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

sess = tf.Session()

# start input queue threads
with tf.name_scope('Queues'):
    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

sess.run(tf.global_variables_initializer())

# run session
#mask, deltas, coords, labels = sess.run([gt_mask, gt_deltas, gt_coords, gt_labels], feed_dict={batch_size: p.BATCH_SIZE})
#final_loss = sess.run([train_step],feed_dict={batch_size: p.BATCH_SIZE})


print('Training initiated')
start_time = time.clock()
for i in range(p.NR_ITERATIONS):
    if i % p.PRINT_FREQ == 0:
        # evaluate loss for mini-batch
        final_loss, b1, conf1, class1 = sess.run([total_loss,bbox_loss, confidence_loss, classification_loss], feed_dict={batch_size: p.BATCH_SIZE})
        print("step %d, train loss = %g, time taken = %g seconds" % (i, final_loss, time.clock() - start_time))
        print("Bbox loss = %g, Confidence loss = %g, Class los = %g"%(b1, conf1, class1))
        start_time = time.clock()
    # optimise network
    sess.run(train_step, feed_dict={batch_size: p.BATCH_SIZE})