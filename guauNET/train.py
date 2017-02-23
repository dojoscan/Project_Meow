import tensorflow as tf
import network_function as nf
import input
import loss
import os
import time
import parameters as p

cwd = os.getcwd()

# build input graph
with tf.name_scope('InputPipeline'):
    batch_size = tf.placeholder(dtype=tf.int32, name='BatchSize')
    tf.add_to_collection("BatchSize", batch_size)
    batch = input.create_batch(p.PATH_TO_IMAGES, p.PATH_TO_LABELS, batch_size, p.TRAIN)
    x = batch[0]
    #y_ = tf.one_hot(batch[1], p.NO_CLASSES, dtype=tf.int32, name='OneHot')
    y_=batch[1]
# build CNN graph
network_output = nf.squeeze_net(x)

# build loss graph
cross_entropy, train_accuracy = loss.calculate_loss(network_output, y_)

# build training graph
with tf.name_scope('Training'):
    train_step = tf.train.AdamOptimizer(p.LEARNING_RATE,name='LearningRate').minimize(cross_entropy,name='CrossEntropy')

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

print('Training initiated')
start_time = time.clock()
for i in range(p.NR_ITERATIONS):
    if i % p.PRINT_FREQ == 0:
        # run inference graph
        accuracy, summary = sess.run([train_accuracy, merged_summaries], feed_dict={batch_size: p.BATCH_SIZE})
        print("step %d, train accuracy = %g, time taken = %g seconds" % (i, accuracy, time.clock()-start_time))
        # write accuracy to log file
        summary_writer.add_summary(summary, i)
        start_time = time.clock()
    # run training graph
    sess.run(train_step, feed_dict={batch_size: p.BATCH_SIZE})
print("Final train accuracy = %g" % sess.run(train_accuracy, feed_dict={batch_size: p.BATCH_SIZE}))

# save graph and all parameters as checkpoint
saver.save(sess, p.PATH_TO_CKPT+'\model', global_step=p.NR_ITERATIONS)
print("Model saved in file: %s" % p.PATH_TO_CKPT)