MeowNet: a small convolutional neural network trained for classifcation on the CIFAR10 data. Used primarily to gain a basic
understanding of the effects of operations like normalisation and drop out, and network structure, on training accuracy.

main.py - builds and runs the TF graph in test or train mode
input.py - handles the input pipeline, reading .png images and a single labels txt file
network_function.py - contains the network architectures

Doing a parameter sweep of optimiser settings, ADAM with a learning rate of 0.001 was found to have the highest accuracy
over the mini-batch after 10,000 iterations.  
