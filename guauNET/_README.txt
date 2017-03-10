guauNET: a convolutional neural network-based object detector that processing single images and outputs bounding box predictions. Requires TensorFlow 1.0 and Python 3.5.

test.py - Evaluates a pre-trained network on KITTI
train.py - Trains a network on KITTI
kitti_input.py - Contains the input pipeline; reads the images and labels (masks, deltas etc.) in batches from disk
network.py - TF Graph for the network forward pass
interpretation.py - TF Graph for converting the network output to a more manipulable format
loss.py - TF Graph for calculating the multi-task loss from labels and interpreted network output
filter_prediction.py - TF Graph for filtering the interpreted network output by confidence score and non-maximum suppression
tools.py - Tools for transforming bounding boxes, calculating IOU etc.
parameters.py - Holds all hyperparameters and settings

pre-processing:
	save_labels.py - Creating and saving labels in binary for training data.

Instructions:
1. Ensure that Python 3.5 and TensorFlow 1.0 are installed
2. Download the KITTI training data and labels at http://www.cvlibs.net/datasets/kitti/eval_object.php ('left colour images' and 'training labels')
3. Use pre-processing/set_std_mean.py to compute the mean and standard deviation image of the data set (require PIL).
3. In parameters.py change the paths to direct to
	PATH_TO_IMAGES: folder containing training images
    	PATH_TO_LABELS: folder containing training labels
	[see kitti_input.create_batch for description of the following labels]
    	PATH_TO_DELTAS: folder for storing the deltas
    	PATH_TO_MASK: folder for storing masks
    	PATH_TO_COORDS: folder for storing coordinates
    	PATH_TO_CLASSES: folder for storing class labels
    	PATH_TO_CKPT: folder for storing TF checkpoints
    	PATH_TO_LOGS: folder for storing TF logs
    	PATH_TO_TEST_IMAGES: folder containing test images
    	PATH_TO_TEST_OUTPUT: folder for storing the test predictions
4. Run pre-processing/save_labels to generate the deltas, masks, coordinates, and classes for the training data
5. Run train.py
6. Run test.py

NOTES:
1. Must be atleast one object per training image
	

B. Wu, F. Iandola, P. H. Jin, K. Keutzer, SqueezeDet: Unified, Small, LowPower Fully Convolutional Neural Networks for Real-Time Object Detectionfor Autonomous Driving, in CVPR, 2017.