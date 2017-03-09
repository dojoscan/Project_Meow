guauNET: a convolutional neural network-based object detector that processing single images and outputs bounding box predictions. Requires TensorFlow 1.0 and Python 3.5.

test.py - Evaluates a pre-trained network on KITTI
train.py - Trains a network on KITTI
kitti_input.py - Contains the input pipeline; reads the images and labels (masks, deltas etc.) in batches from disk
network.py - TF Graph for the network forward pass
interpretation.py - TF Graph for converting the network output to a more manipulable format
loss.py - TF Graph for calculating the multi-task loss from labels and interpreted network output
tools.py - Tools for transforming bounding boxes, calculating IOU etc.
parameters.py - Holds all hyperparameters and settings

pre-processing:
	save_labels.py - Creating and saving labels in binary for training data.

Instructions:

B. Wu, F. Iandola, P. H. Jin, K. Keutzer, SqueezeDet: Unified, Small, LowPower Fully Convolutional Neural Networks for Real-Time Object Detectionfor Autonomous Driving, in CVPR, 2017.