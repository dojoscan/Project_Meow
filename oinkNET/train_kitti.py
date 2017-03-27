import tensorflow as tf
import kitti_input as ki
import parameters as p
import network
import interpretation as interp
import loss as l
import os
import time
import numpy as np

batch_size = tf.placeholder(dtype=tf.int32)
keep_prop = tf.placeholder(dtype=tf.float32, name='KeepProp')

# Training
t_batch = ki.create_batch(batch_size, 'Train')
t_image = t_batch[0]
t_mask = t_batch[1]
t_delta = t_batch[2]
t_coord = t_batch[3]
t_class = t_batch[4]

t_network_output, variables_to_save = network.squeeze(t_image, keep_prop)
t_class_scores, t_conf_scores, t_bbox_delta = interp.interpret(t_network_output, batch_size)
t_total_loss, t_bbox_loss, t_conf_loss, t_class_loss, t_l2_loss = l.loss_function\
                        (t_mask, t_delta, t_coord, t_class,  t_bbox_delta, t_conf_scores, t_class_scores, True)
l.add_loss_summaries('Train_', t_total_loss, t_bbox_loss, t_conf_loss, t_class_loss, t_l2_loss)
# build training graph
with tf.variable_scope('Optimisation'):
    global_step = tf.Variable(0, name='GlobalStep', trainable=False)
    train_step = tf.train.AdamOptimizer(p.LEARNING_RATE, name='TrainStep')
    grads_vars = train_step.compute_gradients(t_total_loss, tf.trainable_variables())
    for i, (grad, var) in enumerate(grads_vars):
        grads_vars[i] = (tf.clip_by_value(grad, -10, 10, name='ClippedGradients'), var)
    gradient_op = train_step.apply_gradients(grads_vars, global_step=global_step)

merged_summaries = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(p.PATH_TO_LOGS, graph=tf.get_default_graph())

# saver for creating checkpoints and store specific variables
saver = tf.train.Saver(variables_to_save)
sess = tf.Session()



# start input queue threads
coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

sess.run(tf.global_variables_initializer())

if p.APPLY_TF:
    sess.run(tf.trainable_variables())
    saver.restore(sess, p.PATH_TO_CKPT + '/save/run_simple')
    print("Model restored.")

print('Training initiated')
start_time = time.time()
for i in range(p.NR_ITERATIONS):
    if i % p.PRINT_FREQ == 0:

        # evaluate loss for training mini-batch and apply one step of opt
        t_loss, t_box_l, t_conf_l, t_class_l, summary, _ = sess.run(
            [t_total_loss, t_bbox_loss, t_conf_loss, t_class_loss,
             merged_summaries, gradient_op], feed_dict={batch_size:
                                                            p.BATCH_SIZE, keep_prop: 0.5})
        print("Step %d, total loss = %g, time taken = %g seconds" % (i, t_loss, time.time() - start_time))
        print("Bbox loss = %g, Confidence loss = %g, Class loss = %g" % (t_box_l, t_conf_l, t_class_l))
        print("----------------------------------------------------------------------------")
        # write summaries to log file
        summary_writer.add_summary(summary, global_step=i)
        start_time = time.time()

    # optimise network
    sess.run([gradient_op, global_step], feed_dict={batch_size: p.BATCH_SIZE, keep_prop: 0.5})
if p.APPLY_TF:
    save_path = saver.save(sess, p.PATH_TO_CKPT + '/trans/run_simple')
else:
    save_path = saver.save(sess,p.PATH_TO_CKPT + '/save/run_simple')