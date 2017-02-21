TRAIN = True
LEARNING_RATE = 0.01
NR_ITERATIONS = 10000
PRINT_FREQ = 100
BATCH_SIZE = 128
NO_CLASSES = 10
USER = 'DONAL'
if USER == 'DONAL':
    PATH_TO_IMAGES = "/Users/Donal/Dropbox/CIFAR10/Data/train/images/"
    PATH_TO_LABELS = "/Users/Donal/Dropbox/CIFAR10/Data/train/labels.txt"
    PATH_TO_TEST_IMAGES = "/Users/Donal/Dropbox/CIFAR10/Data/test/images/"
    PATH_TO_LOGS = "/Users/Donal/Desktop/"
    PATH_TO_TEST_OUTPUT = "/Users/Donal/Desktop/"
else:
    PATH_TO_IMAGES = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/train/images/"
    PATH_TO_LABELS = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/train/labels.txt"
    PATH_TO_TEST_IMAGES = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/test/images/"
    PATH_TO_LOGS = "C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/meow_logs/"