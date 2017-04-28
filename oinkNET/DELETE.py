import tensorflow as tf
import imageNET_input as im
import parameters as p
import tools as t
import network
import time

batch_size = tf.placeholder(dtype=tf.int32)

# Training
t_batch = im.create_batch(batch_size=batch_size, mode='Train')
t_lista = t_batch[2]

sess = tf.Session()

# start input queue threads
with tf.variable_scope('Threads'):
    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

print('Training initiated!')
start_time = time.time()
for i in range(0, p.NR_ITERATIONS):

        tlist=sess.run(t_lista, feed_dict={batch_size:p.BATCH_SIZE})
        print(i)
        print("----------------------------------------------------------------------------")

#prim_saver.save(sess, p.PATH_TO_CKPT + 'prim/run', global_step=global_step)
#prim_cont_saver.save(sess, p.PATH_TO_CKPT + 'prim_cont/run', global_step=global_step)
print("Training completed!")