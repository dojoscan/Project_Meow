# TRAIN A CLASSIFICATION NETWORK FROM SCRATCH USING IMAGENET DATASET

import tensorflow as tf
import imageNET_input as im
import parameters as p
import network
import time
import os
import glob

prim_cont_ckpt = tf.train.get_checkpoint_state(p.PATH_TO_CKPT + 'prim_cont/')

batch_size = tf.placeholder(dtype=tf.int32)
keep_prop = tf.placeholder(dtype=tf.float32, name='KeepProp')

# Training
with tf.device("/cpu:0"):
    t_batch = im.create_batch(batch_size=batch_size, mode='Train')
t_image = t_batch[0]
t_class = tf.one_hot(t_batch[1], p.PRIM_NR_CLASSES, dtype=tf.int32)

t_network_output, _ = network.forget_squeeze_net(t_image, keep_prop, False, False)

t_correct_prediction = tf.equal(tf.argmax(t_network_output, 1), tf.argmax(t_class, 1))
t_accuracy = tf.reduce_mean(tf.cast(t_correct_prediction, tf.float32))

# Validation
with tf.device("/cpu:0"):
    v_batch = im.create_batch(batch_size, 'Val')
v_image = v_batch[0]
v_class = tf.one_hot(v_batch[1], p.PRIM_NR_CLASSES, dtype=tf.int32)

v_network_output, _ = network.forget_squeeze_net(v_image, keep_prop, False, True)
v_correct_prediction = tf.equal(tf.argmax(v_network_output, 1), tf.argmax(v_class, 1))
v_accuracy = tf.reduce_mean(tf.cast(v_correct_prediction, tf.float32))

# saver for creating checkpoints and store specific variables
prim_cont_saver = tf.train.Saver()
sess = tf.Session()

# start input queue threads
with tf.variable_scope('Threads'):
    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

#initialize variables
if prim_cont_ckpt:
    #restore_path, init_step = get_ckpt_order(p.PATH_TO_CKPT + 'prim_cont/')
    newest = sorted(glob.iglob(p.PATH_TO_CKPT + 'prim_cont/' + '*.meta'))
    #prim_cont_saver.restore(sess, restore_path)
    #print("Restored from ImageNet trained network. Step %d, Dir = " % init_step + restore_path)
else:
    print("Error.")

# initialize variables and training
print('Checking ImageNet accuracy initiated!')
start_time = time.time()
filename_val= 'Validation_accuracy.txt'
filename_train= 'Training_accuracy.txt'
# 20
for i in range(0,20):
    split_path = newest[i].split('.')[0]
    init_step = int(split_path.split('-')[-1])
    prim_cont_saver.restore(sess, split_path)
    print("Restored from ImageNet trained network. Step %d, Dir = " % init_step + split_path)

    # evaluate loss for validation mini-batch
    val_acc = sess.run(v_accuracy, feed_dict={batch_size: 50000, keep_prop: 0.5})
    # evaluate loss for training mini-batch and apply one step of opt
    train_acc = sess.run(t_accuracy, feed_dict={batch_size: 1300000, keep_prop: 0.5})
    print("Validation Accuracy = %g, Training Accuracy = %g, time taken = %g fps" % (val_acc, train_acc, (50000+1300000)/(time.time() - start_time)))
    print("----------------------------------------------------------------------------")
    place_text_val = os.path.join(p.PATH_TO_ACCURACY, filename_val)
    with open(place_text_val, 'w') as a:
        wr = ('%.2f' % val_acc) + (' ')
        a.write(wr)
        a.close()
    place_text_train = os.path.join(p.PATH_TO_ACCURACY, filename_train)
    with open(place_text_train, 'w') as a:
        wr = ('%.2f' % train_acc) + (' ')
        a.write(wr)
        a.close()
    start_time = time.time()
