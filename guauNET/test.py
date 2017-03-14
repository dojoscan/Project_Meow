# TESTING PIPELINE FOR KITTI

import tensorflow as tf
import network as net
import kitti_input as ki
import parameters as p
import interpretation as interp
import loss as l
import filter_prediction as fp
import os
import time
import numpy as np
cwd = os.getcwd()

# build input graph
with tf.name_scope('InputPipeline'):
    batch_size = tf.placeholder(dtype=tf.int32, name='BatchSize')
    batch = ki.create_batch(batch_size, p.TRAIN)
    x = batch[0]

# build CNN graph
keep_prop = tf.placeholder(dtype = tf.float32, name='KeepProp')
network_output = net.squeeze_net(x, keep_prop)

# build interpretation graph
class_scores, confidence_scores, bbox_delta = interp.interpret(network_output, batch_size)

# build filtering graph
final_boxes, final_probs, final_class = fp.filter(class_scores, confidence_scores, bbox_delta)

saver = tf.train.Saver()
sess = tf.Session()

# restore from checkpoint
saver.restore(sess, p.PATH_TO_CKPT_TEST)

# start queues
coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

# run testing
start_time=time.clock()
sum_time=0
for i in range(0,int(round(p.NR_OF_TEST_IMAGES/p.TEST_BATCH_SIZE))):
    fbox, fprobs, fclass = sess.run([final_boxes, final_probs, final_class], feed_dict={batch_size: p.TEST_BATCH_SIZE, keep_prop: 1})
    # Write labels
    fp.write_labels(fbox, fclass, fprobs, (i*p.TEST_BATCH_SIZE))
    print("step %d, time taken = %g seconds"%(i, time.clock()-start_time))
    sum_time += time.clock()-start_time
    start_time=time.clock()

print("average time taken per image = %g seconds" % (sum_time/p.NR_OF_TEST_IMAGES))