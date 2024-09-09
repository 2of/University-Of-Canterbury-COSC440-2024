import pickle
import numpy as np
import tensorflow as tf
import os

def unpickle(file):
	"""
	CIFAR data contains the files data_batch_1, data_batch_2, ...,
	as well as test_batch. We have combined all train batches into one
	batch for you. Each of these files is a Python "pickled"
	object produced with cPickle. The code below will open up a
	"pickled" object (each file) and return a dictionary.

	NOTE: DO NOT EDIT

	:param file: the file to unpickle
	:return: dictionary of unpickled data
	"""
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def pre_process_data(inputs, labels, first_class, second_class):
	"""
	Given two ndarrays of inputs and labels and two target classes, returns an array of
	normalized inputs (images) and an array of labels.
	You will want to first extract only the data that matches the
	corresponding classes we want (there are 10 classes and we only want 2).
	You should make sure to normalize all inputs and also turn the labels
	into one hot vectors using tf.one_hot().
	Note that because you are using tf.one_hot() for your labels, your
	labels will be a Tensor, while your inputs will be a NumPy array. This
	is fine because TensorFlow works with NumPy arrays.
	:param file_path: file path for inputs and labels, something
	like 'CIFAR_data_compressed/train'
	:param first_class:  an integer (0-9) representing the first target
	class in the CIFAR10 dataset, for a cat, this would be a 3
	:param first_class:  an integer (0-9) representing the second target
	class in the CIFAR10 dataset, for a dog, this would be a 5
	:return: normalized NumPy array of inputs and tensor of labels, where
	inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels
	has size (num_examples, num_classes)
	"""
	# TODO: Do the preprocessing!
	pass
	mask = np.isin(labels, [first_class, second_class])
	filtered_inputs = inputs[mask]
	filtered_labels = labels[mask]
	filtered_inputs = np.float32(filtered_inputs) / 255

	filtered_labels = np.where(filtered_labels == first_class, 0 , 1)
	ohlabels = tf.one_hot(filtered_labels, depth = 2)

	reshaped_inputs = np.reshape(filtered_inputs, (-1, 3, 32, 32))
	reshaped_inputs = np.transpose(reshaped_inputs, (0, 2, 3, 1))

	# print(ohlabels[0:19])
	#print(reshaped_inputs.shape, ohlabels.shape)
	return reshaped_inputs,ohlabels

def get_data(file_path, first_class, second_class):
	"""
	Given a file path and two target classes, returns an array of
	normalized inputs (images) and an array of labels.

	:param file_path: file path for inputs and labels, something
	like 'CIFAR_data_compressed/train'
	:param first_class:  an integer (0-9) representing the first target
	class in the CIFAR10 dataset, for a cat, this would be a 3
	:param first_class:  an integer (0-9) representing the second target
	class in the CIFAR10 dataset, for a dog, this would be a 5
	:return: normalized NumPy array of inputs and tensor of labels, where
	inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels
	has size (num_examples, num_classes)
	"""
	unpickled_file = unpickle(file_path)
	inputs = unpickled_file[b'data']
	labels = unpickled_file[b'labels']

	return pre_process_data(inputs, labels, first_class, second_class)
