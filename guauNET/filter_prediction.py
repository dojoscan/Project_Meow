# SELECT TOP K BOXES PER IMAGE AND APPLY NMS

import tensorflow as tf
import parameters as p
import loss as l
import os

def find_k_best(det_probs, det_boxes, det_class):
    "' Find the k best boxes with better probabilities." \
    "Args: " \
    "   det_probs, det_boxes, det_class" \
    "Returns:" \
    "   probs: a 2d tensor with the k best probabilities, sz=[batch_sz, no_anchors_per_image]" \
    "   boxes: a 3d tensor with the bounding boxes coordinates corresponding to the k best prob, sz=[batch_sz, no_anchors_per_image,4]" \
    "   class_index: a 2d tensor with the classes of the object from the k best prob, sz=[batch_sz, no_anchors_per_image]"
    probs, index = tf.nn.top_k(det_probs, k=p.NR_TOP_DETECTIONS, sorted=True, name=None)
    index = tf.reshape(index, [p.TEST_BATCH_SIZE, p.NR_TOP_DETECTIONS])
    boxes = []
    class_index = []
    for i in range(0, p.TEST_BATCH_SIZE):
        boxes.append(tf.gather(det_boxes[i, :, :], index[i, :]))
        class_index.append(tf.gather(det_class[i, :], index[i, :]))
    boxes = tf.reshape(boxes, [p.TEST_BATCH_SIZE, p.NR_TOP_DETECTIONS, 4])
    class_index = tf.reshape(class_index, [p.TEST_BATCH_SIZE, p.NR_TOP_DETECTIONS])
    return probs, boxes, class_index

def nms(probs, boxes, class_index):
    "'Apply non maximum supressiont to the boxes." \
    "Args:" \
    "   probs, boxes, class_index" \
    "Return: " \
    "   final_boxes, final_class, final_probs "
    final_boxes = []
    final_probs = []
    final_class = []
    for i in range(0, p.TEST_BATCH_SIZE):
        boxes_image = tf.reshape(boxes[i, :, :], [-1, 4])
        probs_image = tf.reshape(probs[i, :], [-1])
        class_image = tf.reshape(class_index[i, :], [-1])
        coun = 0
        for j in range(0, p.NR_CLASSES):
            class_bool = tf.equal(class_image, tf.constant(j, dtype=tf.int32))
            class_indx = tf.cast(tf.where(class_bool), dtype=tf.int32)
            boxes_class = tf.reshape(tf.gather(boxes_image, class_indx), [-1, 4])
            probs_class = tf.reshape(tf.gather(probs_image, class_indx), [-1])
            if probs_class is not None:
                idx_nms = tf.image.non_max_suppression(boxes_class, probs_class, p.NR_TOP_DETECTIONS, iou_threshold=p.NMS_THRESHOLD,
                                                       name='NMS')
                idx = tf.reshape(tf.gather(class_indx, idx_nms), [-1])
                # if idx is not None:
                if coun == 0:
                    final_idx = idx
                else:
                    final_idx = tf.concat([final_idx, idx], 0)
                coun += 1

        final_boxes.append(tf.gather(boxes_image, final_idx))
        final_probs.append(tf.gather(probs_image, final_idx))
        final_class.append(tf.gather(class_image, final_idx))
    return final_boxes, final_probs, final_class

def filter(class_scores, confidence_scores, bbox_delta):
    "' Calculates the bounding boxes and their corresponding confidence score (probability) and class, from the CNN" \
    "output after applying non maximum supression (NMS) to the k first boxes with better 'probability'. This last term is " \
    "calculated multiplying the classification score and the confidence scores." \
    "Args:" \
    "   class_scores: a 3d tensor containing the Conf(Cl|Obj) dist. for each anchor [batch_sz, no_anchors_per_image, no_classes]" \
    "   confidence_scores: a 2d tensor containing the Conf(Obj)*IOU for each anchor [batch_sz, no_anchors_per_image]" \
    "   bbox_delta: a 3d tensor containing the parameterised offsets for each anchor [batch_sz, no_anchors_per_image,4]" \
    "Returns:" \
    "   final_boxes: a 1d tensor containing other tensors with the xmin, ymin, xmax, ymax of the bboxes selected after nms per image]" \
    "   final_probs: a 1d tensor containing other tensor with the probabilities for the bboxes selected after nms per image" \
    "   final_class: a 1d tensor containing other tensors with the classes from the bbox selected after nms per image"
    # calcualtion of probabilities
    probs_per_class = tf.multiply(class_scores,
                                  tf.reshape(confidence_scores, [p.TEST_BATCH_SIZE, p.NR_ANCHORS_PER_IMAGE, 1]))
    det_probs = tf.reduce_max(probs_per_class,
                              axis=2)  # search for the maximum number (per image per anchor) along the classes
    det_class = tf.cast(tf.argmax(probs_per_class, axis=2),
                        tf.int32)  # gives the index of the class of the maximum number
    det_boxes = l.transform_deltas_to_bbox(bbox_delta, False)

    # find k best
    probs, boxes, class_index = find_k_best(det_probs, det_boxes, det_class)

    # nms
    final_boxes, final_probs, final_class = nms(probs, boxes, class_index)

    return final_boxes, final_probs, final_class

def write_labels(fbox, fclass, fprobs, index):

    for i in range(0, len(fbox)):
        nr_objects = len(fclass[i])
        filename = ('%06d' % (i+index)) + '.txt'
        place_text = os.path.join(p.PATH_TO_WRITE_LABELS, filename)
        with open(place_text, 'w') as a:
            for j in range(0, nr_objects):
                # np.savetxt(place_text, fclass[i][j])
                wr = p.CLASSES_INV[('%s' % fclass[i][j])] + (' ') + ('%s' % -1) + (' ') + ('%s' % -1) + (' ') + \
                     ('%s' % -10) + (' ') + ('%.2f' % fbox[i][j, 0]) + (' ') + ('%.2f' % fbox[i][j, 1]) + (' ') +\
                     ('%.2f' % fbox[i][j, 2]) + (' ') + ('%.2f' % fbox[i][j, 3]) + (' ') + ('%s %s %s' % (-1, -1, -1)) +\
                     (' ') + ('%s %s %s' % (-1000, -1000, -1000)) + (' ') + ('%s' % -10) + (' ') +\
                     ('%.2f' % fprobs[i][j]) + ('\n')
                a.write(wr)
            a.close()
