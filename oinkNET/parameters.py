import numpy as np

TRAIN = True
APPLY_TF=False
KEEP_PROP=0.5
IM_SIZE = 32
LEARNING_RATE = 0.001
NR_ITERATIONS = 500
PRINT_FREQ = 100
BATCH_SIZE = 128
NO_CLASSES = 10

USER = 'LUCIA'

if USER == 'DONAL':
    PATH_TO_DATA = '/Users/Donal/Dropbox/KITTI/data/'
    PATH_TO_OUTPUT = '/Users/Donal/Desktop/output/'
elif USER == 'LUCIA':
    # cifar10 data
    PATH_TO_DATA = 'C:/Master Chalmers/2 year/volvo thesis/code0/MEOW/Data/'
    PATH_TO_OUTPUT = 'C:/log_ckpt_thesis/transfer_learning/'
elif USER == 'BILL':
    PATH_TO_DATA = "/Users/LDIEGO/Documents/KITTI/data/"
    PATH_TO_OUTPUT = "/Users/LDIEGO/Documents/KITTI/output/"
else:
    PATH_TO_DATA = "/home/ad-tool-wd-1/Documents/DONALLUCIA/KITTIdata/"
    PATH_TO_OUTPUT = "/home/ad-tool-wd-1/Documents/DONALLUCIA/Output/"

# training
PATH_TO_IMAGES = PATH_TO_DATA + "train/images/"
PATH_TO_LABELS = PATH_TO_DATA + "train/labels.txt"
PATH_TO_LOGS = PATH_TO_OUTPUT + "logs/"
PATH_TO_CKPT = PATH_TO_OUTPUT + "ckpt/"

# testing
PATH_TO_TEST_IMAGES = PATH_TO_DATA + "test/images/"