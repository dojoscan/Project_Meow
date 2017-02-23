TRAIN = True
LEARNING_RATE = 0.001
NR_ITERATIONS = 100
PRINT_FREQ = 100
BATCH_SIZE = 128
NO_CLASSES = 9
IMAGE_WIDTH = 414
IMAGE_HEIGHT = 125
NR_ANCHOR_PER_CELL=9
classes = {'Car': '0', 'Van': '1','Truck': '2', 'Pedestrian': '3', 'Person_sitting': '4','Cyclist': '5','Tram': '6','Misc': '7','DontCare': '8'}
USER = 'LUCIA'
if USER == 'DONAL':
    PATH_TO_IMAGES = "/Users/Donal/Dropbox/KITTI/test/image/"
    PATH_TO_LABELS = "/Users/Donal/Dropbox/KITTI/test/label/"
    PATH_TO_TEST_IMAGES = "/Users/Donal/Dropbox/CIFAR10/Data/test/images/"
    PATH_TO_LOGS = "/Users/Donal/Desktop/"
    PATH_TO_TEST_OUTPUT = "/Users/Donal/Desktop/"
    PATH_TO_CKPT = "/Users/Donal/Dropbox/KITTI/checkpoints/"
else:
    #PATH_TO_IMAGES = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/train/images/"
    PATH_TO_IMAGES="C:/Master Chalmers/2 year/volvo thesis/code0/test/image/"
    #PATH_TO_LABELS = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/train/labels.txt"
    PATH_TO_LABELS="C:/Master Chalmers/2 year/volvo thesis/code0/test/label/"
    PATH_TO_TEST_IMAGES = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/test/images/"
    PATH_TO_LOGS = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/meow_logs/"
    PATH_TO_TEST_OUTPUT = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/test_output/"
    PATH_TO_CKPT = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/checkpoints/"