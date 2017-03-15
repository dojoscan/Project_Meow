import tensorflow as tf
import parameters as p
from PIL import Image
import numpy as np

filename = tf.placeholder(tf.string)
file_contents = tf.read_file(filename)
image = tf.image.decode_png(file_contents, channels=3, name='Image')
resize_im = tf.image.resize_images(image, [p.IMAGE_HEIGHT, p.IMAGE_WIDTH])
norm_image = tf.divide(tf.subtract(resize_im,p.MEAN_IMAGE), p.STD_IMAGE)
maxi = tf.reduce_max(norm_image)
mini = tf.reduce_min(norm_image)
rescale_image = tf.round(tf.multiply(tf.divide(tf.subtract(norm_image,mini),(maxi-mini)),255))

sess = tf.Session()
n_im = sess.run(rescale_image, feed_dict={filename:'/Users/Donal/Dropbox/KITTI/test/image/000011.png'})

im = Image.fromarray(np.uint8(p.MEAN_IMAGE),'RGB')
im.save("/Users/Donal/Desktop/norm_image11.jpeg")

