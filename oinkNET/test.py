# TESTING PIPELINE FOR KITTI

import tensorflow as tf
import network
import kitti_input as ki
import parameters as p
import interpretation as interp
import filter_prediction as fp
import tools as t
import time

batch_size = tf.placeholder(dtype=tf.int32)
keep_prop = tf.placeholder(dtype=tf.float32, name='KeepProp')

# Testing
with tf.device("/cpu:0"):
    batch = ki.create_batch(batch_size=batch_size, mode='Train')
image = batch[0]

# CNN graph
network_output, _ = network.squeeze(image, keep_prop, True)
class_scores, conf_scores, bbox_delta = interp.interpret(network_output, batch_size)

# build filtering graph
final_boxes, final_probs, final_class = fp.filter(class_scores, conf_scores, bbox_delta)

test_saver = tf.train.Saver()
sess = tf.Session()

# start input queue threads
with tf.variable_scope('Threads'):
    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

# restore from checkpoint
restore_path, _ = t.get_last_ckpt(p.PATH_TO_CKPT + 'sec/')
test_saver.restore(sess, restore_path)
print("Restored from ImageNet and KITTI trained network. Ready for testing. Dir = " + restore_path)

# run testing
start_time = time.time()
sum_time = 0
for i in range(0, int(round(p.NR_OF_TEST_IMAGES/p.TEST_BATCH_SIZE))):
    image, fbox, fprobs, fclass, net_out = sess.run([image, final_boxes, final_probs, final_class, network_output],
                                                    feed_dict={batch_size: p.TEST_BATCH_SIZE, keep_prop: 1})
    # Write labels
    fp.write_labels(fbox, fclass, fprobs, (i*p.TEST_BATCH_SIZE))
    print("Batch %d, Processing speed = %g fps" % (i, i*p.TEST_BATCH_SIZE/(time.time()-start_time)))
    sum_time += time.time()-start_time
    start_time = time.time()

print("average time taken per image = %g seconds" % (sum_time/p.NR_OF_TEST_IMAGES))