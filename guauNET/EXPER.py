import tensorflow as tf
import network as net
import kitti_input as ki
import parameters as p
import interpretation as interp
import loss as l
import time

# build input graph
with tf.name_scope('InputPipeline'):
    batch_size = tf.placeholder(dtype=tf.int32, name='BatchSize')
    batch = ki.create_batch(batch_size, train=True)
    x = batch[0]
    gt_mask = batch[1]
    gt_deltas = batch[2]
    gt_coords = batch[3]
    gt_labels = batch[4]

bbox = l.transform_deltas_to_bbox(gt_deltas, True)

sess = tf.Session()

# start input queue threads
with tf.name_scope('Queues'):
    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

sess.run(tf.global_variables_initializer())

for i in range(0, 4):
    BOX, MASK, LABELS, COORDS = sess.run([bbox, gt_mask, gt_labels, gt_coords], feed_dict={batch_size:p.BATCH_SIZE})

import numpy as np
Box = np.squeeze(BOX)
Mask = np.squeeze(MASK)
Label = np.squeeze(LABELS)
Coords = np.squeeze(COORDS)
print(np.sum(Mask))
print('-----------------------')
print(Box[Mask==1,:])
print('----------------------')
print(Label[Mask==1,:])
print('----------------------')
print(Coords[Mask==1,:])
print('----------------------')

