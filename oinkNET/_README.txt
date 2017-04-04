oinkNET: a convolutional neural network-based object detector that processing single images and outputs bounding box predictions. Requires TensorFlow 1.0 and Python 3.5.
This implementation is similair to guauNET but allows for transfer learning from a classification task to a detection task

prim_train.py - Pre-training on a classification task
sec_train.py - Training a pretrained network on a detection task
imageNET_input.py - Reading and preprocessing ImageNet data

NOTES:
1. ImageNet 2012 recognition challenge data can be found at http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads

B. Wu, F. Iandola, P. H. Jin, K. Keutzer, SqueezeDet: Unified, Small, LowPower Fully Convolutional Neural Networks for Real-Time Object Detectionfor Autonomous Driving, in CVPR, 2017.