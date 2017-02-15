import tensorflow as tf
import network_function as nf
import input
import os
import time

# SUMMARY
# BIG RUN
# DROPOUT
# DIFFERENT OPTIMISER

TRAIN = True
LEARNING_RATE = 0.0001
NR_ITERATIONS = 10000
PRINT_FREQ = 100
BATCH_SIZE = 128
NO_CLASSES = 10
PATH_TO_IMAGES = "/Users/Donal/Dropbox/CIFAR10/Data/train/images/"
PATH_TO_LABELS = "/Users/Donal/Dropbox/CIFAR10/Data/train/labels.txt"
PATH_TO_TEST_IMAGES="/Users/Donal/Dropbox/CIFAR10/Data/test/images/"
PATH_TO_LOGS = "/Users/Donal/Dropbox/CIFAR10/optimizer accuracy/"
#PATH_TO_IMAGES = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/train/images/"
#PATH_TO_LABELS ="C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/train/labels.txt"
#PATH_TO_TEST_IMAGES="C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/test/images/"
cwd = os.getcwd()

if TRAIN:
    batch = input.create_batch(PATH_TO_IMAGES, PATH_TO_LABELS, BATCH_SIZE, TRAIN)
    x = batch[0]
    y_ = tf.one_hot(batch[1], NO_CLASSES, dtype=tf.int32)

    # build CNN graph
    h_pool3 = nf.meow_net(x)

    # build training graph
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(h_pool3, y_))
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(h_pool3, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    summary_op = tf.summary.scalar("Accuracy", accuracy)

    saver = tf.train.Saver()
    sess = tf.Session()

    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(PATH_TO_LOGS, graph=tf.get_default_graph())

    # run training
    print('Training initiated')
    start_time = time.clock()
    for i in range(NR_ITERATIONS):
        if i % PRINT_FREQ == 0:
            train_accuracy, summary = sess.run([accuracy,summary_op])
            print("step %d, train accuracy = %g, time taken = %g seconds" % (i, train_accuracy, time.clock()-start_time))
            summary_writer.add_summary(summary, i)
            start_time = time.clock()
        sess.run(train_step)

    print("Final train accuracy = %g" % sess.run(accuracy))
    save_path = saver.save(sess, cwd + "/checkpoint/meow_run_0.ckpt")

else:
    # build test graph
    batch = input.create_batch(PATH_TO_TEST_IMAGES, PATH_TO_LABELS, 349, False)
    x = batch[0]
    h_pool3 = nf.meow_net(x)
    class_prob = tf.nn.softmax(h_pool3)

    saver = tf.train.Saver()
    sess = tf.Session()

    # restore from checkpoint
    saver.restore(sess, cwd + "/checkpoint/meow_run_0.ckpt")
    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)
    sess.run(tf.global_variables_initializer())
    #writer = tf.train.SummaryWriter("output", sess.graph)

    # run testing
    p_c = sess.run(class_prob)

