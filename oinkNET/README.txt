oinkNET: a convolutional neural network-based object detector that processing single images and outputs bounding box predictions. Requires TensorFlow 1.0 and Python 3.5.
This implementation is similair to guauNET but allows for transfer learning from a classification task to a detection task

prim_train.py - Pre-training on a classification task
sec_train.py - Training a pretrained network on a detection task
imageNET_input.py - Reading and preprocessing ImageNet data

NOTES:
1. ImageNet 2012 recognition challenge data can be found at http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads

Instructions:
1. Ensure that Python 3.5 and TensorFlow 1.0 are installed
2. Download the KITTI data and labels at http://www.cvlibs.net/datasets/kitti/eval_object.php ('left colour images' and 'training labels')
, create the training labels and split into training and validation sets
3. Download the ImageNet training and validation data at http://www.image-net.org/download-images
4. Pre-train network using prim_train.py
5. Train for object detection using sec_train.py

B. Wu, F. Iandola, P. H. Jin, K. Keutzer, SqueezeDet: Unified, Small, LowPower Fully Convolutional Neural Networks for Real-Time Object Detectionfor Autonomous Driving, in CVPR, 2017.