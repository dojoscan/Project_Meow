import numpy as np
TRAIN = True
LEARNING_RATE = 0.001
EPSILON = 0.0001
NR_ITERATIONS = 100
PRINT_FREQ = 100
BATCH_SIZE = 20
NR_CLASSES = 9
IMAGE_WIDTH = 1242
IMAGE_HEIGHT = 375
OUTPUT_WIDTH = 25
OUTPUT_HEIGHT = 6
NR_ANCHORS_PER_CELL = 9
CLASSES = {'Car': '0', 'Van': '1', 'Truck': '2', 'Pedestrian': '3', 'Person_sitting': '4', 'Cyclist': '5', 'Tram': '6', 'Misc': '7', 'DontCare': '8'}
USER = 'LUCIA'
if USER == 'DONAL':
    PATH_TO_IMAGES = "/Users/Donal/Dropbox/KITTI/test/image/"
    PATH_TO_LABELS = "/Users/Donal/Dropbox/KITTI/test/label/"
    PATH_TO_TEST_IMAGES = "/Users/Donal/Dropbox/CIFAR10/Data/test/images/"
    PATH_TO_LOGS = "/Users/Donal/Desktop/"
    PATH_TO_TEST_OUTPUT = "/Users/Donal/Desktop/"
    PATH_TO_CKPT = "/Users/Donal/Dropbox/KITTI/checkpoints/"
    PATH_TO_DELTAS = "/Users/Donal/Dropbox/KITTI/test/deltas/"
else:
    #PATH_TO_IMAGES = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/train/images/"
    PATH_TO_IMAGES="C:/Master Chalmers/2 year/volvo thesis/code0/test/image/"
    #PATH_TO_LABELS = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/train/labels.txt"
    PATH_TO_LABELS="C:/Master Chalmers/2 year/volvo thesis/code0/test/label/"
    PATH_TO_TEST_IMAGES = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/test/images/"
    PATH_TO_LOGS = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/meow_logs/"
    PATH_TO_TEST_OUTPUT = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/test_output/"
    PATH_TO_CKPT = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/checkpoints/"
    PATH_TO_DELTAS = "C:/Master Chalmers/2 year/volvo thesis/code0/test/deltas/"
    PATH_TO_MASK = "C:/Master Chalmers/2 year/volvo thesis/code0/test/mask/"
    PATH_TO_COORDS = "C:/Master Chalmers/2 year/volvo thesis/code0/test/coords/"
    PATH_TO_CLASSES = "C:/Master Chalmers/2 year/volvo thesis/code0/test/classes/"

def set_anchors():
  H, W, B = 22, 76, 9
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
