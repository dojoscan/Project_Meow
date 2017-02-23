import tensorflow as tf
import os
import glob
import useful_functions as uf

IM_SIZE = 32

options={'Car':'0', 'Van':'1','Truck':'2', 'Pedestrian':'3', 'Person_sitting':'4','Cyclist':'5','Tram':'6','Misc':'7','DontCare':'8'}

def read_image(filename):
    """
    Args:
        filename: a scalar string tensor.
    Returns:
        image_tensor: decoded image
    """

    file_contents = tf.read_file(filename)
    image = tf.image.decode_png(file_contents, channels=3)
    image = tf.image.resize_images(image, [IM_SIZE, IM_SIZE])
    return image

def read_labels(path_to_labels):
    """
    Args:
        path_to_labels: full path to labels file
    Returns:
        labels: a 1D tensor of all labels
        bbox:
    """
    #labels=[]
    label_list=os.listdir(path_to_labels)
    #idx=000000
    labels=[]
    for file in [path_to_labels + s for s in label_list]:
        data = open(file, 'r').read()
        labels.append(data)
        #for i in open(file,'r'):
            #obj=i.strip().split(' ')
            #obj[0]=options[obj[0]]
            #print(obj)


     #   bboxes=[]
     #   for i in data:
      #      obj = i.strip().split(' ')
      #      cls=obj[0] #class: pedestrian, cyclist, car,..
       #     xmin = float(obj[4])
        #    ymin = float(obj[5])
         #   xmax = float(obj[6])
          #  ymax = float(obj[7])
            #x, y, w, h = uf.bbox_transform_inv([xmin, ymin, xmax, ymax])
            #bboxes.append([x, y, w, h, cls])
        #labels[idx] = bboxes
        #idx=idx+1
    return labels

def create_image_list(path_to_images):
    """
    Args:
        path_to_images: full path to image folder
    Returns:
        image_list: a tensor of all files in that folder
    """

    image_list = os.listdir(path_to_images)
    image_list = [path_to_images + s for s in image_list]
    return image_list

def create_batch(path_to_images, path_to_labels, batch_size, train):
    """
    Args:
        path_to_images: full path to input images folder
        path_to_labels: full path to input labels
        batch_size: number of examples in mini-batch
        train: boolean for training or testing mode
    Returns:
        batch: list of images as a 4d tensor sz = [batch_sz, im_h, im_w, im_d]
        and labels as a 1d tensor sz = [batch_sz]
    """
    image_list = create_image_list(path_to_images)
    no_samples = len(image_list)
    print(no_samples)
    image_list = tf.convert_to_tensor(image_list, dtype=tf.string)
    if train:
        labels = read_labels(path_to_labels)
    else:   # Create fake labels for testing data
        labels = [0]*no_samples
    labels = tf.convert_to_tensor(labels,dtype=tf.string)
    input_queue = tf.train.slice_input_producer([image_list, labels], shuffle=False)
    images = read_image(input_queue[0])
    batch = tf.train.batch([images, input_queue[1]], batch_size=batch_size)
    return batch

