MeowNet: a small convolutional neural network trained for classifcation on the CIFAR10 data. Used primarily to gain a basic
understanding of the affects of operations like normalisation and drop out, and network structure, on training accuracy.

main.py - builds and runs the TF graph in test or train mode
input.py - handles the input pipeline, reading .png images and a single labels txt file
network_function.py - contains the network architectures

Instructions:
1. Make sure Python 3.5 and TensorFlow 1.0 are installed.
2. Download the training data at https://www.dropbox.com/sh/2je7jyhogqcfozh/AAAb8QFJluRCyBkALc_-b01-a?dl=0 (CIFAR10 as .png files and one .txt for labels)
3. Change the paths declared at the start of main.py
	For training
	# PATH_TO_IMAGES -> folder contain the downloaded image
	# PATH_TO_LABELS -> full path to labels .txt file
	# PATH_TO_LOGS -> a folder for storing the TF logs
	# PATH_TO_CKPT -> a folder for storing the TF checkpoints
	For testing
	# PATH_TO_TEST_IMAGES -> folder contain any test images (of size 32x32x3)
	# PATH_TO_TEST_OUPTUT -> full path to text file for writing the network's class predictions
4. Run training with Train = True
5. Test its performance on the test data with Train = False