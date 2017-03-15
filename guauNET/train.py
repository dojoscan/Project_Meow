# TRAINING PIPELINE FOR KITTI

import tensorflow as tf
import network as net
import kitti_input as ki
import parameters as p
import interpretation as interp
import loss as l
import os
import time
import numpy as np

cwd = os.getcwd()

ckpt = tf.train.get_checkpoint_state(p.PATH_TO_CKPT)

# build input graph
batch_size = tf.placeholder(dtype=tf.int32, name='BatchSize')
batch = ki.create_batch(batch_size, True)
x = batch[0]
gt_mask = batch[1]
gt_deltas = batch[2]
gt_coords = batch[3]
gt_labels = batch[4]

# build CNN graph
keep_prop = tf.placeholder(dtype=tf.float32, name='KeepProp')
network_output = net.asym_squeeze_net(x, keep_prop)

# build interpretation graph
class_scores, confidence_scores, bbox_delta = interp.interpret(network_output, batch_size)

# build loss graph
total_loss, bbox_loss, confidence_loss, classification_loss, l2_loss = l.loss_function(gt_mask, gt_deltas, gt_coords,  bbox_delta
                                                                                              , confidence_scores, gt_labels, class_scores, True)

# build training graph
with tf.variable_scope('Optimisation'):
    global_step = tf.Variable(0, name='GlobalStep', trainable=False)
    learning_rate = tf.train.exponential_decay(p.LEARNING_RATE, global_step,
                                           p.DECAY_STEP, p.DECAY_FACTOR, staircase=True, name='LearningRate')
    train_step = tf.train.AdamOptimizer(learning_rate, name='TrainStep')
    grads_vars = train_step.compute_gradients(total_loss, tf.trainable_variables())
    for i, (grad, var) in enumerate(grads_vars):
        grads_vars[i] = (tf.clip_by_norm(grad, 1, name='ClippedGradients'), var)
    gradient_op = train_step.apply_gradients(grads_vars, global_step=global_step)

# summaries for TensorBoard
tf.summary.scalar('Total_loss', total_loss)
tf.summary.scalar('Bounding_box_loss', bbox_loss)
tf.summary.scalar('Object_confidence_loss', confidence_loss)
tf.summary.scalar('Classification_loss', classification_loss)
tf.summary.scalar('Weight_decay_loss', l2_loss)
tf.summary.scalar('Learning_rate', learning_rate)
merged_summaries = tf.summary.merge_all()

# saver for creating checkpoints
saver = tf.train.Saver(name='Saver')
sess = tf.Session()

# start input queue threads
with tf.variable_scope('Threads'):
    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

# initialise variables
if ckpt:
    restore_path = tf.train.latest_checkpoint(p.PATH_TO_CKPT)
    saver.restore(sess, restore_path)
    init_step = int(restore_path.split('/')[-1].split('-')[-1])
else:
    sess.run(tf.global_variables_initializer())
    init_step = 0
summary_writer = tf.summary.FileWriter(p.PATH_TO_LOGS, graph=tf.get_default_graph())

# training
print('Training initiated')
start_time = time.clock()
for i in range(init_step, p.NR_ITERATIONS):
     if i % p.CKPT_FREQ == 0:
         # save learnable variables in checkpoint
        saver.save(sess, p.PATH_TO_CKPT + 'run', global_step=global_step)
        print("Step %d checkpoint saved at " % i + p.PATH_TO_CKPT)
        print("----------------------------------------------------------------------------")
     if i % p.PRINT_FREQ == 0:
         # evaluate loss for mini-batch
         final_loss, b1, conf1, class1, summary, _ = sess.run([total_loss, bbox_loss, confidence_loss, classification_loss, merged_summaries, gradient_op],
                                                             feed_dict={batch_size: p.BATCH_SIZE, keep_prop: 0.5})
         print("Step %d, total loss = %g, time taken = %g seconds" % (i, final_loss, time.clock() - start_time))
         print("Bbox loss = %g, Confidence loss = %g, Class loss = %g" % (b1, conf1, class1))
         print("----------------------------------------------------------------------------")
         # write accuracy to log file
         summary_writer.add_summary(summary, global_step=i)
         start_time = time.clock()
     else:
        _,_ = sess.run([gradient_op, global_step], feed_dict={batch_size: p.BATCH_SIZE, keep_prop: 0.5})


print("Training completed!")