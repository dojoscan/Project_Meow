# TESTING PIPELINE FOR KITTI (NO PRE-TRAINING)

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
    with tf.device("/cpu:0"):
        batch = ki.create_batch(batch_size, 'Test')
    x = batch[0]
    input_filename = batch[5]

tf.summary.image('InputImage', x, max_outputs=5)

# build CNN graph
keep_prop = tf.placeholder(dtype=tf.float32, name='KeepProp')
network_output = net.res_squeeze_net(x, keep_prop, False)

# build interpretation graph
class_scores, confidence_scores, bbox_delta = interp.interpret(network_output, batch_size)

# build filtering graph
final_boxes, final_probs, final_class = fp.filter(class_scores, confidence_scores, bbox_delta)

saver = tf.train.Saver()
sess = tf.Session()

# restore variables from checkpoint
restore_path, _ = t.get_last_ckpt(p.PATH_TO_CKPT_TEST)
saver.restore(sess, restore_path)
print('Restored variables from ' + restore_path)

# start queues
coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

merged_summaries = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(p.PATH_TO_LOGS, graph=tf.get_default_graph())

# run testing
start_time = time.time()
sum_time = 0
for i in range(0, int(round(p.NR_OF_TEST_IMAGES/p.TEST_BATCH_SIZE))):
    summ, fbox, fprobs, fclass, image_id = sess.run([merged_summaries, final_boxes, final_probs, final_class, input_filename],
                                                    feed_dict={batch_size: p.TEST_BATCH_SIZE, keep_prop: 1})
    summary_writer.add_summary(summ)
    # Write labels in KITTI format
    fp.write_labels(fbox, fclass, fprobs, image_id)
    print("Batch %d, Processing speed = %g fps" % (i, p.TEST_BATCH_SIZE/(time.time()-start_time)))
    sum_time += time.time()-start_time
    start_time = time.time()

print("Average time taken per image = %g seconds" % (sum_time/p.NR_OF_TEST_IMAGES))