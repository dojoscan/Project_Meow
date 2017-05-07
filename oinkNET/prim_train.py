# TRAIN A CLASSIFICATION NETWORK FROM SCRATCH USING IMAGENET DATASET

import tensorflow as tf
import imageNET_input as im
import parameters as p
import tools as t
import network
import time

prim_ckpt = tf.train.get_checkpoint_state(p.PATH_TO_CKPT + 'prim/')
prim_cont_ckpt = tf.train.get_checkpoint_state(p.PATH_TO_CKPT + 'prim_cont/')

batch_size = tf.placeholder(dtype=tf.int32)
keep_prop = tf.placeholder(dtype=tf.float32, name='KeepProp')

# Training
with tf.device("/cpu:0"):
    t_batch = im.create_batch(batch_size=batch_size, mode='Train')
t_image = t_batch[0]
t_class = tf.one_hot(t_batch[1], p.PRIM_NR_CLASSES, dtype=tf.int32)

t_network_output, variables_to_save = network.forget_squeeze_net(t_image, keep_prop, False, False)
t_correct_prediction = tf.equal(tf.argmax(t_network_output, 1), tf.argmax(t_class, 1))
t_accuracy = tf.reduce_mean(tf.cast(t_correct_prediction, tf.float32))
tf.summary.scalar('T_accuracy', t_accuracy)
t_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=t_network_output, labels=t_class))
t_l2_loss = p.WEIGHT_DECAY_FACTOR * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                                                        if 'Bias' not in v.name])
t_total_loss = t_cross_entropy + t_l2_loss
tf.summary.merge([tf.summary.scalar('T_cross_entropy', t_cross_entropy), tf.summary.scalar('T_L2_loss', t_l2_loss),
                  tf.summary.scalar('T_total_loss', t_total_loss)], name='T_loss_summary')
merged_summaries = tf.summary.merge_all()

# build training graph
with tf.variable_scope('Optimisation'):
    global_step = tf.Variable(0, name='GlobalStep', trainable=False)
    train_step = tf.train.AdamOptimizer(p.LEARNING_RATE, name='TrainStep')
    grads_vars = train_step.compute_gradients(t_total_loss, tf.trainable_variables())
    for i, (grad, var) in enumerate(grads_vars):
        grads_vars[i] = (tf.clip_by_value(grad, -100, 100, name='ClippedGradients'), var)
    gradient_op = train_step.apply_gradients(grads_vars, global_step=global_step)

summary_writer = tf.summary.FileWriter(p.PATH_TO_LOGS, graph=tf.get_default_graph())

# Validation
with tf.device("/cpu:0"):
    v_batch = im.create_batch(batch_size, 'Val')
v_image = v_batch[0]
v_class = tf.one_hot(v_batch[1], p.PRIM_NR_CLASSES, dtype=tf.int32)

v_network_output, _ = network.forget_squeeze_net(v_image, keep_prop, False, True)
v_correct_prediction = tf.equal(tf.argmax(v_network_output, 1), tf.argmax(v_class, 1))
v_accuracy = tf.reduce_mean(tf.cast(v_correct_prediction, tf.float32))
v_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=v_network_output, labels=v_class))
val_summ = tf.summary.merge([tf.summary.scalar('V_cross_entropy', v_cross_entropy),
                             tf.summary.scalar('V_accuracy', v_accuracy)], name='V_loss_summaries')

# saver for creating checkpoints and store specific variables
prim_saver = tf.train.Saver(variables_to_save)
prim_cont_saver = tf.train.Saver()
sess = tf.Session()

# start input queue threads
with tf.variable_scope('Threads'):
    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

# initialise variables
if prim_cont_ckpt:
    restore_path, init_step = t.get_last_ckpt(p.PATH_TO_CKPT + 'prim_cont/')
    prim_cont_saver.restore(sess, restore_path)
    print("Restored from ImageNet trained network. Step %d, Dir = " % init_step + restore_path)
else:
    sess.run(tf.global_variables_initializer())
    init_step = 0
    print("Initialised with Xavier weights and zero bias.")

# training
print('Training initiated!')
start_time = time.time()
for i in range(init_step, p.NR_ITERATIONS):
    if i % p.CKPT_FREQ == 0 and i != init_step:
        # save trainable variables in checkpoint
        prim_cont_saver.save(sess, p.PATH_TO_CKPT + 'prim_cont/run', global_step=global_step)
        print("Step %d checkpoint saved at " % i + p.PATH_TO_CKPT + 'prim_cont/')
        print("----------------------------------------------------------------------------")
    if i % p.PRINT_FREQ == 0:
        # evaluate loss for validation mini-batch
        v_cross, val_summary, v_acc = sess.run([v_cross_entropy, val_summ, v_accuracy],
                                               feed_dict={batch_size: p.BATCH_SIZE, keep_prop: 0.5})
        summary_writer.add_summary(val_summary, global_step=i)
        # evaluate loss for training mini-batch and apply one step of op
        t_cross, summary, _, t_acc = sess.run([t_cross_entropy, merged_summaries, gradient_op, t_accuracy],
                                              feed_dict={batch_size:p.BATCH_SIZE, keep_prop: 0.5})
        print("Step %d, Speed = %g fps" % (i, p.PRINT_FREQ *p.BATCH_SIZE/(time.time() - start_time)))
        print("Cross entropy: train = %g, val = %g" % (t_cross, v_cross))
        print("Accuracy: train = %g, val = %g" % (t_acc, v_acc))
        print("----------------------------------------------------------------------------")
        # write summaries to log file
        summary_writer.add_summary(summary, global_step=i)
        start_time = time.time()
    else:
        # apply one step of opt
        sess.run([gradient_op, global_step], feed_dict={batch_size: p.BATCH_SIZE, keep_prop: 0.5})

prim_saver.save(sess, p.PATH_TO_CKPT + 'prim/run', global_step=global_step)
prim_cont_saver.save(sess, p.PATH_TO_CKPT + 'prim_cont/run', global_step=global_step)
print("Training completed!")
