import tensorflow as tf
classes = {'Car':'0', 'Van':'1','Truck':'2', 'Pedestrian':'3', 'Person_sitting':'4','Cyclist':'5','Tram':'6','Misc':'7','DontCare':'8'}

def bbox_transform_inv(bbox):
    """ Convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
    Args:
        bbox: an array
    Returns:
        outbox: an array
    """
    with tf.variable_scope('bbox_transform_inv') as scope:
        xmin, ymin, xmax, ymax = bbox
        out_box = [[]]*4

        width       = xmax - xmin + 1.0
        height      = ymax - ymin + 1.0
        out_box[0]  = xmin + 0.5*width
        out_box[1]  = ymin + 0.5*height
        out_box[2]  = width
        out_box[3]  = height
    return out_box

def separate_labels(y_):
    labels = {}
    index = 0
    for i in y_:
        line = i.split('\n')
        bboxes = []
        for j in line:
            obj = j.split(' ')
            cls = classes[obj[0]]
            xmin = float(obj[4])
            ymin = float(obj[5])
            xmax = float(obj[6])
            ymax = float(obj[7])
            x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
            bboxes.append([x, y, w, h, cls])
        labels[index] = bboxes
        index = index+1
    return labels

def calculate_loss(network_output,y_):
    """ Calculate multi-task loss as described in squeezeDet
    Args:
        network_output: a 4d tensor
        y_: a 1d tensor where each element is a string containing annotations for all objects in a single image
    Returns:
        loss: multi-task loss over a mini-natch
    """
    with tf.name_scope('Loss'):
        #lala
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network_output, labels=y_),
                                       name='CrossEntropy')
        tf.summary.scalar("Cross Entropy", cross_entropy)

        return cross_entropy