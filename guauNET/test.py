# TESTING PIPELINE FOR KITTI

import tensorflow as tf
import network as net
import kitti_input as ki
import parameters as p
import interpretation as interp
import loss as l
import os
import time
import numpy as np
cwd = os.getcwd()

# build input graph
with tf.name_scope('InputPipeline'):
    batch_size = tf.placeholder(dtype=tf.int32, name='BatchSize')
    batch = ki.create_batch(batch_size, p.TRAIN)
    x = batch[0]
    gt_mask = batch[1]
    gt_deltas = batch[2]
    gt_coords = batch[3]
    gt_labels = batch[4]

# build CNN graph
keep_prop = tf.placeholder(dtype = tf.float32, name='KeepProp')
network_output = net.squeeze_net(x, keep_prop)

# build interpretation graph
class_scores, confidence_scores, bbox_delta = interp.interpret(network_output)

#
probs_per_class=tf.multiply(class_scores, tf.reshape(confidence_scores, [p.BATCH_SIZE, p.NR_ANCHORS_PER_IMAGE,1]))

det_probs=tf.reduce_max(probs_per_class, axis=2) # search for the maximum number (per image per anchor) along the classes
det_class=tf.cast(tf.argmax(probs_per_class, axis=2) , tf.int32) # gives the index of the class of the maximum number
det_boxes=l.transform_deltas_to_bbox(bbox_delta)


# find k best
probs, index=tf.nn.top_k(det_probs, k=p.NR_TOP_DETECTIONS, sorted=True, name=None)
index = tf.reshape(index, [p.BATCH_SIZE, p.NR_TOP_DETECTIONS])
boxes=[]
class_index=[]
for i in range(0, p.BATCH_SIZE):
    boxes.append(tf.gather(det_boxes[i,:,:],index[i,:]))
    class_index.append(tf.gather(det_class[i,:], index[i,:]))

# nms
idx_nms=[]
boxes=tf.reshape(boxes,[p.BATCH_SIZE, p.NR_TOP_DETECTIONS, 4])
class_index=tf.reshape(class_index,[p.BATCH_SIZE, p.NR_TOP_DETECTIONS])
final_boxes=[]
final_probs=[]
final_class=[]
for i in range(0, p.BATCH_SIZE):
    boxes_nms=tf.reshape(boxes[i,:,:],[-1,4])
    probs_nms=tf.reshape(probs[i,:],[-1])
    class_nms=tf.reshape(class_index[i,:],[-1])
    for j in range(0,p.NR_CLASSES):
        idx_per_class = [k for k in range(0,p.NR_TOP_DETECTIONS) if class_nms[k] == j]
        print(idx_per_class)
        #idx_nms.append(tf.gather(boxes_nms,idx_per_class))
        #idx_nms.append(tf.image.non_max_suppression(boxes_nms[idx_per_class,:], probs_nms[idx_per_class], p.NR_TOP_DETECTIONS, iou_threshold=None, name=None))
    #final_boxes.append(tf.gather(boxes_nms,idx_nms[i]))
    #final_probs.append(tf.gather(probs_nms, idx_nms[i]))
#final_boxes=tf.reshape(final_boxes,[p.BATCH_SIZE,-1,4])
#final_probs = tf.reshape(final_probs,[p.BATCH_SIZE,-1])


#saver = tf.train.Saver()
sess = tf.Session()

# restore from checkpoint
#saver.restore(sess, p.PATH_TO_CKPT)

# start queues
coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)
sess.run(tf.global_variables_initializer())

# run testing
val, pro, box, idx, cls= sess.run([boxes, class_index, probs, idx_nms, idx_per_class], feed_dict={batch_size: p.BATCH_SIZE, keep_prop: 1})
print(val)
print(pro)
print(box)
print(idx)
print(cls)



#output_file = open(p.PATH_TO_TEST_OUTPUT, 'w')
#np.savetxt(output_file, p_c)