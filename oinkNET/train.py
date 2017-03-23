import tensorflow as tf
import input
import parameters as p
import network
import os
import time
import numpy as np

# build input graph
batch_size = tf.placeholder(dtype=tf.int32)
batch = input.create_batch(p.PATH_TO_IMAGES, p.PATH_TO_LABELS, batch_size, p.TRAIN)
x = batch[0]
gt_labels = tf.one_hot(batch[1], p.NO_CLASSES, dtype=tf.int32)

# build CNN graph:
keep_prop = tf.placeholder(dtype=tf.float32, name='KeepProp')
#h_end, variables_to_save, salvado = network.simple_net(x)
#h_end, variables_to_save, salvado1, salvado2, salvado3, salvado4 = network.deeper_net(x)
h_end, variables_to_save=network.squeeze(x, keep_prop)
print(variables_to_save)

# build training graph
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_end, labels=gt_labels))
train_step = tf.train.AdamOptimizer(p.LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(h_end, 1), tf.argmax(gt_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# saver for creating checkpoints and store specific variables
sess = tf.Session()
saver = tf.train.Saver(variables_to_save)

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
        # evaluate forward pass for mini-batch
        train_accuracy = sess.run([accuracy], feed_dict={batch_size: p.BATCH_SIZE, keep_prop: p.KEEP_PROP})
        print("step %d, train accuracy = %g, time taken = %g seconds" % (i, train_accuracy, time.time()-start_time))
        print('--------------------------------------------------------------------------------------------------')

        start_time = time.time()

    # optimise network
    sess.run(train_step, feed_dict={batch_size: p.BATCH_SIZE, keep_prop: p.KEEP_PROP})
if p.APPLY_TF:
    save_path = saver.save(sess, p.PATH_TO_CKPT + '/trans/run_simple')
else:
    save_path = saver.save(sess,p.PATH_TO_CKPT + '/save/run_simple')
