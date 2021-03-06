# CONTAINS ALL PARAMETERS AND SETTINGS
import numpy as np

# Training
DATA_AUGMENT = True
BATCH_SIZE = 5
PRINT_FREQ = 50
CKPT_FREQ = 1000

# Optimisation
LEARNING_RATE = 0.0001
NR_ITERATIONS = 100000

# PRIMARY (ImageNet)
PRIM_NR_CLASSES = 6
PRIM_IMAGE_WIDTH = 256
PRIM_IMAGE_HEIGHT = 256

# SECONDARY (KITTI)
# Testing
NR_OF_TEST_IMAGES = 50
TEST_BATCH_SIZE = 1
NR_TOP_DETECTIONS = 64
NMS_THRESHOLD = 0.4

# Loss
LAMBDA_BBOX = 5.0
LAMBDA_CONF_POS = 75.0
LAMBDA_CONF_NEG = 100.0
WEIGHT_DECAY_FACTOR = 1E-4

# Network Input
SEC_NR_CLASSES = 3
CLASSES = {'Car': '0', 'Pedestrian': '1', 'Cyclist': '2', 'Van': '3', 'Truck': '4', 'Person_sitting': '5', 'Tram': '6',
           'Misc': '7', 'DontCare': '8'}
CLASSES_INV = {'0': 'Car', '1': 'Pedestrian', '2': 'Cyclist'}
SEC_IMAGE_WIDTH = 1242
SEC_IMAGE_HEIGHT = 375

# Network Output
OUTPUT_WIDTH = 76
OUTPUT_HEIGHT = 22
NR_ANCHORS_PER_CELL = 9

USER = 'LUCIA'

if USER == 'DONAL':
    PATH_TO_DATA = '/Users/Donal/Dropbox/KITTI/data/'
    PATH_TO_OUTPUT = '/Users/Donal/Desktop/output/'
    PATH_TO_PRIM_DATA = '/Users/Donal/Desktop/Thesis/Data/TinyImageNet/'
elif USER == 'LUCIA':
    PATH_TO_DATA = '/Master Chalmers/2 year/volvo thesis/code0/'
    PATH_TO_OUTPUT = 'C:/log_ckpt_thesis/lil_henrik/'
    PATH_TO_PRIM_DATA = 'C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/'
elif USER == 'BILL':
    PATH_TO_DATA = "/Users/LDIEGO/Documents/KITTI/data/"
    PATH_TO_OUTPUT = "/Users/LDIEGO/Documents/KITTI/output/"
    PATH_TO_PRIM_DATA = '/Users/Donal/Dropbox/KITTI/data/'
elif USER == 'LIL HENRIK':
    PATH_TO_DATA = "/Users/ADTOOL-2/Documents/DONALLUCIA/KITTIdata/pre_train/"
    PATH_TO_OUTPUT = "/Users/ADTOOL-2/Documents/DONALLUCIA/Output/"
    PATH_TO_PRIM_DATA = "/Users/ADTOOL-2/Documents/DONALLUCIA/ImageNet/"
else:
    PATH_TO_DATA = "/home/ad-tool-wd-1/Documents/DONALLUCIA/KITTIdata/"
    PATH_TO_OUTPUT = "/home/ad-tool-wd-1/Documents/DONALLUCIA/Output/"
    PATH_TO_PRIM_DATA = '/Users/Donal/Dropbox/KITTI/data/'

# training
PATH_TO_IMAGES = PATH_TO_DATA + "training/image/"
PATH_TO_LABELS = PATH_TO_DATA + "training/label/"
PATH_TO_DELTAS = PATH_TO_DATA + "training/delta/"
PATH_TO_MASK = PATH_TO_DATA + "training/mask/"
PATH_TO_COORDS = PATH_TO_DATA + "training/coord/"
PATH_TO_CLASSES = PATH_TO_DATA + "training/class/"
PATH_TO_LOGS = PATH_TO_OUTPUT + "logs/"
PATH_TO_CKPT = PATH_TO_OUTPUT + "ckpt/"
# testing
PATH_TO_TEST_IMAGES = PATH_TO_DATA + "training/image/"
PATH_TO_CKPT_TEST = PATH_TO_OUTPUT + "ckpt/pre/sec/gated_half/run-10000"
PATH_TO_WRITE_LABELS = PATH_TO_OUTPUT + "predictions/"
# validation
PATH_TO_VAL_IMAGES = PATH_TO_DATA + "validation/image/"
PATH_TO_VAL_DELTAS = PATH_TO_DATA + "validation/delta/"
PATH_TO_VAL_MASK = PATH_TO_DATA + "validation/mask/"
PATH_TO_VAL_COORDS = PATH_TO_DATA + "validation/coord/"
PATH_TO_VAL_CLASSES = PATH_TO_DATA + "validation/class/"


def set_anchors():
    # Dimensions of anchors copied from original SqueezeDet implementation for KITTI
    H, W, B = OUTPUT_HEIGHT, OUTPUT_WIDTH, NR_ANCHORS_PER_CELL
    anchor_shapes = np.reshape(
      [np.array(
          [[36.,  37.], [366., 174.], [115.,  59.],
           [162.,  87.], [38.,  90.], [258., 173.],
           [224., 108.], [78., 170.], [72.,  43.]])] * H * W,
      (H, W, B, 2)
    )
    center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(SEC_IMAGE_WIDTH)/(W+1)]*H*B),
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
    )
    center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(SEC_IMAGE_HEIGHT)/(H+1)]*W*B),
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

NUM_THREADS = 4