# TRAIN A NETWORK ON KITTI THAT HAS BEEN PRE-TRAINED ON IMAGENET

import tensorflow as tf
import kitti_input as ki
import parameters as p
import network
import interpretation as interp
import loss as l
import tools as t
import time

sec_ckpt = tf.train.get_checkpoint_state(p.PATH_TO_CKPT + 'sec/')
prim_ckpt = tf.train.get_checkpoint_state(p.PATH_TO_CKPT + 'prim/')

batch_size = tf.placeholder(dtype=tf.int32)
keep_prop = tf.placeholder(dtype=tf.float32, name='KeepProp')


# Training
with tf.device("/cpu:0"):
    t_batch = ki.create_batch(batch_size=batch_size, mode='Train')
t_image = t_batch[0]
t_mask = t_batch[1]
t_delta = t_batch[2]
t_coord = t_batch[3]
t_class = t_batch[4]

t_network_output, variables_to_save = network.forget_squeeze_net(t_image, keep_prop, True, reuse=False)
t_class_scores, t_conf_scores, t_bbox_delta = interp.interpret(t_network_output, batch_size)
t_total_loss, t_bbox_loss, t_conf_loss, t_class_loss, t_l2_loss = l.loss_function\
                                (t_mask, t_delta, t_coord, t_class,  t_bbox_delta, t_conf_scores, t_class_scores, True)
l.add_loss_summaries('Train_', t_total_loss, t_bbox_loss, t_conf_loss, t_class_loss, t_l2_loss)

# Optimisation
with tf.variable_scope('Optimisation'):
    global_step = tf.Variable(0, name='GlobalStep', trainable=False)
    train_step = tf.train.AdamOptimizer(p.LEARNING_RATE, name='TrainStep')
    grads_vars = train_step.compute_gradients(t_total_loss, tf.trainable_variables())
    for i, (grad, var) in enumerate(grads_vars):
        grads_vars[i] = (tf.clip_by_value(grad, -100, 100, name='ClippedGradients'), var)
    gradient_op = train_step.apply_gradients(grads_vars, global_step=global_step)

merged_summaries = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(p.PATH_TO_LOGS, graph=tf.get_default_graph())

# Validation
with tf.device("/cpu:0"):
    v_batch = ki.create_batch(batch_size, 'Val')
v_image = v_batch[0]
v_mask = v_batch[1]
v_delta = v_batch[2]
v_coord = v_batch[3]
v_class = v_batch[4]

v_network_output, _ = network.forget_squeeze_net(v_image, keep_prop, True, reuse=True)
v_class_scores, v_conf_scores, v_bbox_delta = interp.interpret(v_network_output, batch_size)
v_total_loss, v_bbox_loss, v_conf_loss, v_class_loss, v_l2_loss = l.loss_function\
    (v_mask, v_delta, v_coord, v_class,  v_bbox_delta, v_conf_scores, v_class_scores, True)
val_summ = l.add_loss_summaries('Val_', v_total_loss, v_bbox_loss, v_conf_loss, v_class_loss, v_l2_loss)

# saver for creating checkpoints and store specific variables
prim_saver = tf.train.Saver(variables_to_save, name='PrimSaver')
sec_saver = tf.train.Saver(name='SecSaver', max_to_keep=10)
sess = tf.Session()

# start input queue threads
with tf.variable_scope('Threads'):
    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

# initialise variables
if prim_ckpt and not sec_ckpt:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.trainable_variables())
    restore_path, _ = t.get_last_ckpt(p.PATH_TO_CKPT + 'prim/')
    prim_saver.restore(sess, restore_path)
    init_step = 0
    print("Restored from ImageNet trained network. Dir = " + restore_path)
elif sec_ckpt:
    restore_path, init_step = t.get_last_ckpt(p.PATH_TO_CKPT + 'prim/')
    sec_saver.restore(sess, restore_path)
    print("Restored from ImageNet and KITTI trained network. Step %d, dir = " % init_step + restore_path)
else:
    print("No checkpoints found.")
    print('No ImageNet trained network. Dir = ' + p.PATH_TO_CKPT + 'prim/')
    print('No KITTI trained network. Dir = ' + p.PATH_TO_CKPT + 'sec/')
    exit()

# training
print('Training initiated!')
start_time = time.time()
for i in range(init_step, p.NR_ITERATIONS):
    if i % p.CKPT_FREQ == 0 and i != init_step:
        # save trainable variables in checkpoint
        sec_saver.save(sess, p.PATH_TO_CKPT + 'sec/run', global_step=global_step)
        print("Step %d checkpoint saved at " % i + p.PATH_TO_CKPT + 'sec/')
        print("----------------------------------------------------------------------------")
    if i % p.PRINT_FREQ == 0:
        # evaluate loss for validation mini-batch
        val_summary = sess.run(val_summ, feed_dict={batch_size: p.BATCH_SIZE, keep_prop: 0.5})
        summary_writer.add_summary(val_summary, global_step=i)
        # evaluate loss for training mini-batch and apply one step of opt
        t_loss, t_box_l, t_conf_l, t_class_l, summary, _ = sess.run([t_total_loss, t_bbox_loss, t_conf_loss, t_class_loss,
                                                                merged_summaries, gradient_op], feed_dict={batch_size:
                                                                p.BATCH_SIZE, keep_prop: 0.5})
        print("Step %d, Total loss = %g, Speed = %g fps" % (i, t_loss, p.PRINT_FREQ *
                                                            p.BATCH_SIZE/(time.time() - start_time)))
        print("Bbox loss = %g, Confidence loss = %g, Class loss = %g" % (t_box_l, t_conf_l, t_class_l))
        print("----------------------------------------------------------------------------")
        # write summaries to log file
        summary_writer.add_summary(summary, global_step=i)
        start_time = time.time()
    else:
        # apply one step of opt
        sess.run([gradient_op, global_step], feed_dict={batch_size: p.BATCH_SIZE, keep_prop: 0.5})

sec_saver.save(sess, p.PATH_TO_CKPT + 'sec/run', global_step=global_step)
print("Training completed!")
