import tensorflow as tf
import network_function as nf
import input
import os
import time
import parameters as p

cwd = os.getcwd()

# build input graph
with tf.name_scope('InputPipeline'):
    batch_size = tf.placeholder(dtype=tf.int32, name='BatchSize')
    batch = input.create_batch(p.PATH_TO_IMAGES, p.PATH_TO_LABELS, batch_size, p.TRAIN)
    x = batch[0]
    y_ = tf.one_hot(batch[1], p.NO_CLASSES, dtype=tf.int32, name='OneHot')

# build CNN graph
feature_maps = nf.squeeze_net(x)

# build training graph
with tf.name_scope('Training'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=feature_maps, labels=y_), name='CrossEntropy')
    train_step = tf.train.AdamOptimizer(p.LEARNING_RATE).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(feature_maps, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Accuracy')
    tf.summary.scalar("Training Accuracy", accuracy)
    tf.summary.scalar("Cross Entropy", cross_entropy)

merged_summaries = tf.summary.merge_all()

# saver for creating checkpoints
saver = tf.train.Saver(name='Saver')
sess = tf.Session()

# start input queue threads
with tf.name_scope('Queues'):
    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(p.PATH_TO_LOGS, graph=tf.get_default_graph())

# training
print('Training initiated')
start_time = time.clock()
for i in range(p.NR_ITERATIONS):
    if i % p.PRINT_FREQ == 0:
        # run inference graph
        train_accuracy, summary = sess.run([accuracy, merged_summaries], feed_dict={batch_size: p.BATCH_SIZE})
        print("step %d, train accuracy = %g, time taken = %g seconds" % (i, train_accuracy, time.clock()-start_time))
        # write accuracy to log file
        summary_writer.add_summary(summary, i)
        start_time = time.clock()
    # run training graph
    sess.run(train_step, feed_dict={batch_size: p.BATCH_SIZE})

print("Final train accuracy = %g" % sess.run(accuracy, feed_dict={batch_size: p.BATCH_SIZE}))
save_path = saver.save(sess, cwd + "/checkpoint/meow_run_0.ckpt")
print("Model saved in file: %s" % cwd+"/checkpoint/meow_run_0.ckpt")