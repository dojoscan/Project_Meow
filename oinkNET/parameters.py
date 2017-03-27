import numpy as np

TRAIN = True
APPLY_TF=True
KEEP_PROP=0.5
#IM_SIZE = 32
LEARNING_RATE = 0.001
NR_ITERATIONS = 500
PRINT_FREQ = 10
BATCH_SIZE = 4
#NO_CLASSES = 10

DATA_AUGMENT = True
# Testing
NR_OF_TEST_IMAGES = 7518
TEST_BATCH_SIZE = 3
NR_TOP_DETECTIONS = 64
NMS_THRESHOLD = 0.05

# Loss
LAMBDA_BBOX = 5
LAMBDA_CONF_POS = 75
LAMBDA_CONF_NEG = 100
WEIGHT_DECAY_FACTOR = 0

# Network Input
NR_CLASSES = 9
CLASSES = {'Car': '0', 'Van': '1', 'Truck': '2', 'Pedestrian': '3', 'Person_sitting': '4', 'Cyclist': '5', 'Tram': '6', 'Misc': '7', 'DontCare': '8'}
CLASSES_INV = {'0': 'Car', '1': 'Van', '2': 'Truck', '3': 'Pedestrian', '4': 'Person_sitting', '5': 'Cyclist', '6': 'Tram', '7': 'Misc', '8': 'DontCare'}
IMAGE_WIDTH = 1242
IMAGE_HEIGHT = 375

# Network Output
OUTPUT_WIDTH = 76
OUTPUT_HEIGHT = 22
NR_ANCHORS_PER_CELL = 9


USER = 'LUCIA'

if USER == 'DONAL':
    PATH_TO_DATA = '/Users/Donal/Dropbox/KITTI/data/'
    PATH_TO_OUTPUT = '/Users/Donal/Desktop/output/'
elif USER == 'LUCIA':
    # cifar10 data
    PATH_TO_DATA = 'C:/Master Chalmers/2 year/volvo thesis/code0/test/'
    #PATH_TO_DATA = 'C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/'
    PATH_TO_OUTPUT = 'C:/log_ckpt_thesis/transfer_learning/'
elif USER == 'BILL':
    PATH_TO_DATA = "/Users/LDIEGO/Documents/KITTI/data/"
    PATH_TO_OUTPUT = "/Users/LDIEGO/Documents/KITTI/output/"
else:
    PATH_TO_DATA = "/home/ad-tool-wd-1/Documents/DONALLUCIA/KITTIdata/"
    PATH_TO_OUTPUT = "/home/ad-tool-wd-1/Documents/DONALLUCIA/Output/"

# training
#PATH_TO_IMAGES = PATH_TO_DATA + "train/images/"
#PATH_TO_LABELS = PATH_TO_DATA + "train/labels.txt"
PATH_TO_IMAGES = PATH_TO_DATA + "image/"
PATH_TO_MASK= PATH_TO_DATA + "mask/"
PATH_TO_DELTAS=PATH_TO_DATA + "deltas/"
PATH_TO_COORDS=PATH_TO_DATA + "coords/"
PATH_TO_CLASSES=PATH_TO_DATA + "classes/"
PATH_TO_LOGS = PATH_TO_OUTPUT + "logs/"
PATH_TO_CKPT = PATH_TO_OUTPUT + "ckpt/"

# testing
PATH_TO_TEST_IMAGES = PATH_TO_DATA + "test/images/"


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

EPSILON = 0.0001
