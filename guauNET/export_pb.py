import tensorflow as tf
import network as net
import tools as t
from tensorflow.python.framework import graph_util

PATH_TO_MODEL_CKPT = '/Users/Donal/Desktop/output/ckpt/'
EXPORT_PATH = '/Users/Donal/Desktop/output/model/'
NAME = 'forget_squeeze_70k'


image = tf.placeholder(dtype=tf.float32, name='InputImage')
keep_prop = tf.placeholder(dtype=tf.float32, name='KeepProp')
network_output = net.forget_squeeze_net(image, keep_prop, False)

saver = tf.train.Saver()
sess = tf.Session()

# restore variables from checkpoint
restore_path, _ = t.get_last_ckpt(PATH_TO_MODEL_CKPT)
saver.restore(sess, restore_path)
print('Restored variables from ' + restore_path)

# start queues
coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

# export model
with sess.as_default():
    print('Exporting trained model to ' + EXPORT_PATH)
    graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['CNN/Conv2/AddBias'])
    tf.train.write_graph(graph, EXPORT_PATH, '%s.pb' % NAME, as_text=False)
