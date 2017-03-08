# CONTAINS ALL PARAMETERS AND SETTINGS

import numpy as np

TRAIN = True
DATA_AUGMENT = True
LEARNING_RATE = 0.01
DECAY_STEP = 1000
DECAY_FACTOR = 0.5
EPSILON = 0.0001
NR_ITERATIONS = 5000
PRINT_FREQ = 100
BATCH_SIZE = 2
NR_CLASSES = 9
IMAGE_WIDTH = 1242
IMAGE_HEIGHT = 375
OUTPUT_WIDTH = 76
OUTPUT_HEIGHT = 22
LAMBDA_BBOX = 5
LAMBDA_CONF_POS = 75
LAMBDA_CONF_NEG = 100
NR_ANCHORS_PER_CELL = 9
CLASSES = {'Car': '0', 'Van': '1', 'Truck': '2', 'Pedestrian': '3', 'Person_sitting': '4', 'Cyclist': '5', 'Tram': '6', 'Misc': '7', 'DontCare': '8'}
USER = 'LUCIA'

# UPDATE WITH ACTUAL MEAN AND STD IMAGE
MEAN_IMAGE = np.load('C:/Master Chalmers/2 year/volvo thesis/code0/test/KITTI_mean.txt')
STD_IMAGE = np.load('C:/Master Chalmers/2 year/volvo thesis/code0/test/KITTI_std.txt')
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if USER == 'DONAL':
    PATH_TO_IMAGES = "/Users/Donal/Dropbox/KITTI/test/image/"
    PATH_TO_LABELS = "/Users/Donal/Dropbox/KITTI/test/label/"
    PATH_TO_TEST_IMAGES = "/Users/Donal/Dropbox/CIFAR10/Data/test/images/"
    PATH_TO_LOGS = "/Users/Donal/Desktop/"
    PATH_TO_TEST_OUTPUT = "/Users/Donal/Desktop/"
    PATH_TO_CKPT = "/Users/Donal/Dropbox/KITTI/checkpoints/"
    PATH_TO_DELTAS = "/Users/Donal/Dropbox/KITTI/test/deltas/"
    PATH_TO_MASK = "/Users/Donal/Dropbox/KITTI/test/mask/"
    PATH_TO_COORDS = "/Users/Donal/Dropbox/KITTI/test/coords/"
    PATH_TO_CLASSES = "/Users/Donal/Dropbox/KITTI/test/classes/"
elif USER == 'LUCIA':
    PATH_TO_IMAGES = "C:/Master Chalmers/2 year/volvo thesis/code0/test/image/"
    PATH_TO_LABELS = "C:/Master Chalmers/2 year/volvo thesis/code0/test/label/"
    PATH_TO_TEST_IMAGES = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/test/images/"
    PATH_TO_LOGS = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/meow_logs/"
    PATH_TO_TEST_OUTPUT = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/test_output/"
    PATH_TO_CKPT = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/checkpoints/"
    PATH_TO_DELTAS = "C:/Master Chalmers/2 year/volvo thesis/code0/test/deltas/"
    PATH_TO_MASK = "C:/Master Chalmers/2 year/volvo thesis/code0/test/mask/"
    PATH_TO_COORDS = "C:/Master Chalmers/2 year/volvo thesis/code0/test/coords/"
    PATH_TO_CLASSES = "C:/Master Chalmers/2 year/volvo thesis/code0/test/classes/"
else:
    PATH_TO_IMAGES = "/Users/LDIEGO/Documents/KITTI/KITTIdata/training/image/"
    PATH_TO_LABELS = "/Users/LDIEGO/Documents/KITTI/KITTIdata/training/label/"
    PATH_TO_DELTAS = "/Users/LDIEGO/Documents/KITTI/KITTIdata/training/delta/"
    PATH_TO_MASK = "/Users/LDIEGO/Documents/KITTI/KITTIdata/training/mask/"
    PATH_TO_COORDS = "/Users/LDIEGO/Documents/KITTI/KITTIdata/training/coords/"
    PATH_TO_CLASSES = "/Users/LDIEGO/Documents/KITTI/KITTIdata/training/class/"
    PATH_TO_CKPT = "/Users/LDIEGO/Documents/KITTI/KITTIdata/training/_output/TFckpt/"
    PATH_TO_LOGS = "/Users/LDIEGO/Documents/KITTI/KITTIdata/training/_output/TFlogs/"
    PATH_TO_TEST_IMAGES = "/Users/LDIEGO/Documents/KITTI/KITTIdata/testing/image/"
    PATH_TO_TEST_OUTPUT = "/Users/LDIEGO/Documents/KITTI/KITTIdata/testing/_output/"

def set_anchors():
  H, W, B = OUTPUT_HEIGHT, OUTPUT_WIDTH, NR_ANCHORS_PER_CELL
  anchor_shapes = np.reshape(
      [np.array(
          [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
           [ 162.,  87.], [  38.,  90.], [ 258., 173.],
           [ 224., 108.], [  78., 170.], [  72.,  43.]])] * H * W,
      (H, W, B, 2)
  )
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(IMAGE_WIDTH)/(W+1)]*H*B),
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return anchors

ANCHORS = set_anchors()
NR_ANCHORS_PER_IMAGE = len(ANCHORS)
