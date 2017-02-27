MeowNet: a small convolutional neural network trained for classifcation on the CIFAR10 data. Used primarily to gain a basic
understanding of the affects of operations like normalisation and drop out, and network structure, on training accuracy.

main.py - builds and runs the TF graph in test or train mode
input.py - handles the input pipeline, reading .png images and a single labels txt file
network_function.py - contains the network architectures

The training data can be downloaded at https://www.dropbox.com/sh/2je7jyhogqcfozh/AAAb8QFJluRCyBkALc_-b01-a?dl=0

Doing a parameter sweep of optimiser settings, ADAM with a learning rate of 0.001 was found to have the highest accuracy
over the mini-batch after 10,000 iterations.
