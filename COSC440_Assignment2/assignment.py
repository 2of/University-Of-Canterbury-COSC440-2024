from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random


'''
FYI 
The indentation in this notebook is borked
used chatgpt to fix before starting part1
'''


#
#python3 -m venv tfenv
#source tfenv/bin/activate
#
#






# Something in this file throws the indentation in VS Code off and it's infuriating!

def linear_unit(x, W, b):
    return tf.matmul(x, W) + b

class ModelPart0:
    def __init__(self):
        """
        This model class contains a single layer network similar to Assignment 1.
        """

        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        input = 32 * 32 * 3
        output = 2
        self.W1 = tf.Variable(tf.random.truncated_normal([input, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B1 = tf.Variable(tf.random.truncated_normal([1, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")

        self.trainable_variables = [self.W1, self.B1]

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)

        # This reshape "flattens" the image data
        
        #print("Shape before", inputs.shape)
        inputs = np.reshape(inputs, [inputs.shape[0], -1])
        
       # print(inputs.shape)
        x = linear_unit(inputs, self.W1, self.B1)
        return x

class ModelPart1:
    def __init__(self):
        """
        This model class contains a single layer network similar to Assignment 1.
        
        
        Add another linear layer by replicating the W1 and B1 to W2 and B2. We’d like to densely connect the two layers, so the size of the output for layer 1 should be 256 and the size of the input for layer 2 should be 256 (and its output size 2).

        """

        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)



        input = 32 * 32 * 3
        layer1_output = 256
        layer2_input = 256
        layer2_output = 2 #classifier
        
        
        
        self.W1 = tf.Variable(tf.random.truncated_normal([input, layer1_output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B1 = tf.Variable(tf.random.truncated_normal([1, layer1_output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")
        
        self.W2 = tf.Variable(tf.random.truncated_normal([layer2_input, layer2_output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W2")
        self.B2 = tf.Variable(tf.random.truncated_normal([1, layer2_output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B2")
	
			

        self.trainable_variables = [self.W1, self.B1,self.W2,self.B2]

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        
        '''
        Edit the code for the forward pass in call so that it calls the 
        ReLU activation function after the first layer and then calls the linear_unit function
        for the second layer.
        Remember that the output from one layer should be the input to the next layer.
        
        
        '''
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # channels is rgb?

        # This reshape "flattens" the image data
       # print("Shape before", inputs.shape)
        inputs = np.reshape(inputs, [inputs.shape[0], -1])
        
        #print(inputs.shape)
        
        l1 = linear_unit(inputs, self.W1,self.B1)
        l1 = tf.nn.relu(l1)
        
        l2 = linear_unit(l1,self.W2,self.B2)
        

        return l2



class ModelPart3:
    def __init__(self):
        """
        This model class contains a multi-layer network with a convolutional layer followed by
        two fully connected layers. The first fully connected layer has an output size of 256, 
        and the second one has an output size of 2 (for classification).
        """

        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        input_shape = 32 * 32 * 3  
        layer1_output = 256  
        layer2_output = 2   
        
		#####
		# filter size & number filters to change
        self.cf_size = 3 
        self.cf_num_filters = 24 #oh no computer slow!
        ####
        
        self.cW = tf.Variable(tf.random.truncated_normal([self.cf_size, self.cf_size, 3, self.cf_num_filters],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="cW")
        self.cB = tf.Variable(tf.random.truncated_normal([self.cf_num_filters],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="cB")
        
 
        input_from_C = 16 * 16 * self.cf_num_filters  
        #see the pooling layer in .call(), not super duper sure, but it works :) 
        
        
        self.W1 = tf.Variable(tf.random.truncated_normal([input_from_C, layer1_output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B1 = tf.Variable(tf.random.truncated_normal([1, layer1_output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")
        
        self.W2 = tf.Variable(tf.random.truncated_normal([layer1_output, layer2_output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W2")
        self.B2 = tf.Variable(tf.random.truncated_normal([1, layer2_output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B2")


        self.trainable_variables = [self.cW, self.cB, self.W1, self.B1, self.W2, self.B2]

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """

	#	lc = linear_unit(tf.nn.conv2d(inputs, self.cW))
        lc = tf.nn.conv2d(inputs, self.cW, strides=[1, 1, 1, 1], padding="SAME") + self.cB
        lc = tf.nn.relu(lc)
        lc = tf.nn.max_pool(lc, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		# This ISNT in the doc / guide / ass2 handout, but I had to do it to get the sizes of the matrices to talk kindly to each other....... 
        lc = tf.reshape(lc, [lc.shape[0], -1])
        l1 = linear_unit(lc, self.W1, self.B1)
        l1 = tf.nn.relu(l1)
        l2 = linear_unit(l1, self.W2, self.B2)

        return l2

class ModelPart3_a:
    def __init__(self):
        """
      extension of q3
        """

        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        input_shape = 32 * 32 * 3  
        layer1_output = 256  
        layer2_output = 2   
        
        # change parts
        self.cf_size = 3
        self.cf_size_l2 = 5
        self.cf_size_l3 = 3
                
        self.cf_num_filters = 32
        self.cf_num_filters_l2 = 64
        self.cf_num_filters_l3 = 64

        self.cW = tf.Variable(tf.random.truncated_normal([self.cf_size, self.cf_size, 3, self.cf_num_filters],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="cW")
        self.cB = tf.Variable(tf.random.truncated_normal([self.cf_num_filters],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="cB")
        
        self.cW1 = tf.Variable(tf.random.truncated_normal([self.cf_size_l2, self.cf_size_l2, self.cf_num_filters, self.cf_num_filters_l2],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="cW1")
        self.cB1 = tf.Variable(tf.random.truncated_normal([self.cf_num_filters_l2],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="cB1")
        
        self.cW2 = tf.Variable(tf.random.truncated_normal([self.cf_size_l3, self.cf_size_l3, self.cf_num_filters_l2, self.cf_num_filters_l3],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="cW2")
        self.cB2 = tf.Variable(tf.random.truncated_normal([self.cf_num_filters_l3],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="cB2")
        
        # UPDATE ME UPDATE ME UPDATE ME
        input_from_C = 4 * 4 * self.cf_num_filters_l3
        
        self.W1 = tf.Variable(tf.random.truncated_normal([input_from_C, layer1_output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B1 = tf.Variable(tf.random.truncated_normal([layer1_output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")
        
        self.W2 = tf.Variable(tf.random.truncated_normal([layer1_output, layer2_output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W2")
        self.B2 = tf.Variable(tf.random.truncated_normal([layer2_output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B2")
        self.trainable_variables = [self.cW, self.cB, self.cW1, self.cB1, self.cW2, self.cB2, self.W1, self.B1, self.W2, self.B2]

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """

        lc = tf.nn.conv2d(inputs, self.cW, strides=[1, 1, 1, 1], padding="SAME") + self.cB
        lc = tf.nn.relu(lc)
        lc = tf.nn.max_pool(lc, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        lc = tf.nn.conv2d(lc, self.cW1, strides=[1, 1, 1, 1], padding="SAME") + self.cB1
        lc = tf.nn.relu(lc)
        lc = tf.nn.max_pool(lc, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        lc = tf.nn.conv2d(lc, self.cW2, strides=[1, 1, 1, 1], padding="SAME") + self.cB2
        lc = tf.nn.relu(lc)
        lc = tf.nn.max_pool(lc, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        

        lc = tf.reshape(lc, [lc.shape[0], -1])
        l1 = linear_unit(lc, self.W1, self.B1)
        l1 = tf.nn.relu(l1)
        l2 = linear_unit(l1, self.W2, self.B2)

        return l2


def linear_unit(x, W, b):
    return tf.matmul(x, W) + b



    
    
def loss(logits, labels):
	"""
	Calculates the cross-entropy loss after one forward pass.
	:param logits: during training, a matrix of shape (batch_size, self.num_classes)
	containing the result of multiple convolution and feed forward layers
	Softmax is applied in this function.
	:param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
	:return: the loss of the model as a Tensor
	"""
	loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
	return tf.reduce_mean(loss)
	

def accuracy(logits, labels):
	"""
	Calculates the model's prediction accuracy by comparing
	logits to correct labels – no need to modify this.
	:param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
	containing the result of multiple convolution and feed forward layers
	:param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

	NOTE: DO NOT EDIT

	:return: the accuracy of the model as a Tensor
	"""
	correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather.
    You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: None
    '''
    indices = tf.random.shuffle([i for i in range(len(train_labels))])
    train_inputs = tf.gather(train_inputs, indices)    
    train_labels = tf.gather(train_labels, indices)
    
    for i in range(0, len(train_inputs), model.batch_size):
        # print(i) yeah so above was the bug !! (Was skipping through images by bunch not by image)
        batch_inputs = train_inputs[i: i + model.batch_size]
        batch_labels = train_labels[i: i + model.batch_size]
        
        with tf.GradientTape() as tape:
            logits = model.call(batch_inputs)
            batch_loss = loss(logits, batch_labels)
        gradients = tape.gradient(batch_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
	"""
	Tests the model on the test inputs and labels.
	:param test_inputs: test data (all images to be tested),
	shape (num_inputs, width, height, num_channels)
	:param test_labels: test labels (all corresponding labels),
	shape (num_labels, num_classes)
	:return: test accuracy - this can be the average accuracy across
	all batches or the sum as long as you eventually divide it by batch_size
	"""
	return accuracy(model.call(test_inputs), test_labels)


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
	"""
	Uses Matplotlib to visualize the results of our model.
	:param image_inputs: image data from get_data(), limited to 10 images, shape (10, 32, 32, 3)
	:param probabilities: the output of model.call(), shape (10, num_classes)
	:param image_labels: the labels from get_data(), shape (10, num_classes)
	:param first_label: the name of the first class, "dog"
	:param second_label: the name of the second class, "cat"

	NOTE: DO NOT EDIT

	:return: doesn't return anything, a plot should pop-up
	"""
	predicted_labels = np.argmax(probabilities, axis=1)
	num_images = image_inputs.shape[0]

	fig, axs = plt.subplots(ncols=num_images)
	fig.suptitle("PL = Predicted Label\nAL = Actual Label")
	for ind, ax in enumerate(axs):
			ax.imshow(image_inputs[ind], cmap="Greys")
			pl = first_label if predicted_labels[ind] == 0.0 else second_label
			al = first_label if np.argmax(image_labels[ind], axis=0) == 0 else second_label
			ax.set(title="PL: {}\nAL: {}".format(pl, al))
			plt.setp(ax.get_xticklabels(), visible=False)
			plt.setp(ax.get_yticklabels(), visible=False)
			ax.tick_params(axis='both', which='both', length=0)
	plt.show()


CLASS_CAT = 3
CLASS_DOG = 5
NUM_EPOCHS = 25 #low for testing


def main(cifar_data_folder):
    train_data, train_labels = get_data(f'{cifar_data_folder}train', CLASS_CAT, CLASS_DOG)
    test_data, test_labels = get_data(f'{cifar_data_folder}test', CLASS_CAT, CLASS_DOG)
    
    model = ModelPart3_a()

    for i in range(NUM_EPOCHS):
        train(model, train_data, train_labels)
        accuracy_value = test(model, test_data, test_labels).numpy()
        print("epoch  # ", i, ", accuracy : ", accuracy_value)
    test_accuracy = test(model, test_data, test_labels).numpy()
    print("Accuracy ovearll :", test_accuracy)



    # vis
    batch_size = 10
    sample_indices = np.random.choice(len(test_data), batch_size, replace=False)
    sample_indices = tf.convert_to_tensor(sample_indices) 
    sample_images = tf.gather(test_data, sample_indices)
    sample_labels = tf.gather(test_labels, sample_indices)
    
    logits = model.call(sample_images)
    probabilities = tf.nn.softmax(logits).numpy()
    
    visualize_results(sample_images.numpy(), probabilities, sample_labels.numpy(), "Cat", "Dog")
if __name__ == '__main__':
    print("sanity check")
    cifar_data_folder = './CIFAR_data/'
    main(cifar_data_folder)
