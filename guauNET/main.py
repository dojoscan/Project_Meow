import tensorflow as tf
import network_function as nf
import input
import os
import time
import parameters as p

cwd = os.getcwd()

# build input graph
batch_size = tf.placeholder(dtype=tf.int32)
batch = input.create_batch(p.PATH_TO_IMAGES, p.PATH_TO_LABELS, batch_size, p.TRAIN)
x = batch[0]
y_ = tf.one_hot(batch[1], p.NO_CLASSES, dtype=tf.int32)

# build CNN graph
h_pool3 = nf.meow_net(x)

if TRAIN:

    # build training graph
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_pool3, labels=y_))
    train_step = tf.train.AdamOptimizer(p.LEARNING_RATE).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(h_pool3, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    summary_op = tf.summary.scalar("Accuracy", accuracy)

    # saver for creating checkpoints
    saver = tf.train.Saver()
    sess = tf.Session()

    # start input queue threads
    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

    sess.run(tf.global_variables_initializer())
    #summary_writer = tf.summary.FileWriter(PATH_TO_LOGS, graph=tf.get_default_graph())

    # training
    print('Training initiated')
    start_time = time.clock()
    for i in range(p.NR_ITERATIONS):
        if i % p.PRINT_FREQ == 0:
            # evaluate forward pass for mini-batch
            train_accuracy, summary = sess.run([accuracy, summary_op], feed_dict={batch_size: p.BATCH_SIZE})
            print("step %d, train accuracy = %g, time taken = %g seconds" % (i, train_accuracy, time.clock()-start_time))
            # write accuracy to log file
            # summary_writer.add_summary(summary, i)
            start_time = time.clock()
        # optimise network
        sess.run(train_step, feed_dict={batch_size: p.BATCH_SIZE})

    print("Final train accuracy = %g" % sess.run(accuracy, feed_dict={batch_size: p.BATCH_SIZE}))
    save_path = saver.save(sess, cwd + "/checkpoint/meow_run_0.ckpt")

else:

    # get prob. dist. over classes
    class_prob = tf.nn.softmax(h_pool3)

    saver = tf.train.Saver()
    sess = tf.Session()

    # restore from checkpoint
    saver.restore(sess, cwd + "/checkpoint/meow_run_0.ckpt")

    # start queues
    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)
    sess.run(tf.global_variables_initializer())

    # run testing
    p_c = sess.run(class_prob, feed_dict={batch_size: 5})
    print(p_c)

