import tensorflow as tf

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [32, 32, 3])
    image = tf.cast(image, tf.float32)
    label = tf.cast(features['label'], tf.int32)
    return image, label

def get_batch(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    image, label = read_and_decode(filename_queue)
    batch = tf.train.batch([image, label], batch_size=batch_size)
    return batch




