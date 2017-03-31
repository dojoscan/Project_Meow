# TESTING PIPELINE FOR KITTI

import tensorflow as tf
import network as net
import kitti_input as ki
import parameters as p
import interpretation as interp
import tools as t
import filter_prediction as fp
import time

# build input graph
with tf.name_scope('InputPipeline'):
    batch_size = tf.placeholder(dtype=tf.int32, name='BatchSize')
    batch = ki.create_batch(batch_size, 'Test')
    x = batch[0]

# build CNN graph
keep_prop = tf.placeholder(dtype=tf.float32, name='KeepProp')
network_output = net.forget_squeeze_net(x, keep_prop)

# build interpretation graph
class_scores, confidence_scores, bbox_delta = interp.interpret(network_output, batch_size)

# build filtering graph
final_boxes, final_probs, final_class = fp.filter(class_scores, confidence_scores, bbox_delta)

saver = tf.train.Saver()
sess = tf.Session()

# restore from checkpoint
restore_path, _ = t.get_last_ckpt(p.PATH_TO_CKPT)
saver.restore(sess, restore_path)

# start queues
coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

# run testing
start_time = time.time()
sum_time = 0
for i in range(0, int(round(p.NR_OF_TEST_IMAGES/p.TEST_BATCH_SIZE))):
    image, fbox, fprobs, fclass, net_out = sess.run([x, final_boxes, final_probs, final_class, network_output], feed_dict={batch_size: p.TEST_BATCH_SIZE, keep_prop: 1})
    # Write labels
    fp.write_labels(fbox, fclass, fprobs, (i*p.TEST_BATCH_SIZE))
    print("Batch %d, Processing speed = %g fps" % (i, i*p.TEST_BATCH_SIZE/(time.time()-start_time)))
    sum_time += time.time()-start_time
    start_time = time.time()

print("Average time taken per image = %g seconds" % (sum_time/p.NR_OF_TEST_IMAGES))