from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
from preprocess import get_data
import gzip, os

class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying MNIST with
    batched learning. Please implement the TODOs for the entire
    model but do not change the method and constructor arguments.
    Make sure that your Model class works with multiple batch
    sizes. Additionally, please exclusively use NumPy and
    Python built-in functions for your implementation.
    """

    def __init__(self):
        # TODO: Initialize all hyperparametrs
        self.input_size = 784 # Size of image vectors
        self.num_classes = 10 # Number of classes/possible labels
        self.batch_size = 100 #doc
        self.learning_rate = 0.5 #doc

        # TODO: Initialize weights and biases
        self.W = np.zeros((self.num_classes, self.input_size))
        self.b = np.zeros((self.num_classes, 1))
        
        
    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
        (batch_size x 784) (2D), where batch can be any number.
        :return: output, unscaled output values for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass 
        return (inputs @ self.W.T) + self.b.T


    def back_propagation(self, inputs, outputs, labels):
        #outputs = probs of pass
        """
        Returns the gradients for model's weights and biases
        after one forward pass. The learning algorithm for updating weights
        and biases is the Perceptron Learning Algorithm discussed in
        lecture (and described in the assignment writeup). This function should
        handle a batch_size number of inputs by taking the average of the gradients
        across all inputs in the batch.
        :param inputs: batch inputs (a batch of images)
        :param outputs: matrix that contains the unscaled output values of each
        class for each image
        :param labels: true labels
        :return: gradient for weights, and gradient for biases
        """
        # TODO: calculate the gradients for the weights and the gradients for the bias with respect to average loss
        # HINT: np.argmax(outputs, axis=1) will give the index of the largest output
      #  print(inputs.shape, outputs.shape, labels.shape)
        pred_classification = np.argmax(outputs, axis=1)
        labels_matrix = np.eye(self.num_classes)[labels]
        max_len = min(self.batch_size, len(pred_classification))
        predicted_class_matrix = np.zeros_like(labels_matrix)
        predicted_class_matrix[np.arange(max_len), pred_classification] = 1
            
        
        # yc per doc
        # there could be a more efficient way to do this tahn the two passes below?
        yc = (labels_matrix * (1 - predicted_class_matrix)) - ((1 - labels_matrix) * predicted_class_matrix)
        

        gradW = np.dot(yc.T, inputs) /  len(labels)
        gradB = np.sum(yc, axis=0).reshape(-1, 1) / len(labels)
        #See above the reshape(-1,1) which I still don't understand WHY this is neceassry but here we are :)
       # print(gradW)
        return gradW, gradB


        #Below: Iterative approach (old) but it does work :)
        pred_classification = np.argmax(outputs, axis=1)
        gradW = np.zeros_like(self.W)
        gradB = np.zeros_like(self.b)

        for i in range(len(labels)):
            #print(i)#
            #bad approach which goes iteratively (on the batch so it's actually not that bad.... yet)
            for c in range(self.num_classes):
                if labels[i] == c and pred_classification[i] != c:
                    y = 1
                elif labels[i] != c and pred_classification[i] == c:
                    y = -1
                else:
                    y = 0
                gradW[c] += y * inputs[i]
                gradB[c] += y
        
        gradW /= len(labels)
        gradB /= len(labels)
        print(gradW.shape, gradB.shape)
        return gradW, gradB

    
    
    def accuracy(self, outputs, labels):
        """
        Calculates the model's accuracy by comparing the number
        of correct predictions with the correct answers.
        :param outputs: result of running model.call() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy
        return np.sum(labels == outputs) / len(outputs)

        

    def gradient_descent(self, gradW, gradB):
        
        """
        Given the gradients for weights and biases, does gradient
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        """
        # TODO: change the weights and biases of the model to descent the gradient
        self.W += gradW * self.learning_rate
        #because adding vectors here ! not vals then add to res
        self.b += gradB * self.learning_rate

        
        

def train(model, train_inputs, train_labels):
    """
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    """

    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    for start in range(0, len(train_inputs), model.batch_size):
      #  print("train # ", start)
        inputs = train_inputs[start:start+model.batch_size]
        labels = train_labels[start:start+model.batch_size]

        # TODO: For every batch, compute then descend the gradients for the model's weights
        probabilities = model.call(inputs)
        # print("Probabilities", probabilities)
        gradientsW, gradientsB = model.back_propagation(inputs, probabilities, labels)
        model.gradient_descent(gradientsW, gradientsB)


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment,
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    :param test_inputs: MNIST test data (all images to be tested)
    :param test_labels: MNIST test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    probs = model.call(test_inputs)
    accuracy = model.accuracy(np.argmax(probs, axis=1), test_labels)
    print( "The overall accuracy on test data is", accuracy)
    return accuracy

    
    # TODO: Return accuracy across testing set

def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.call()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()


def main(mnist_data_folder):
    """
    Read in MNIST data, initialize your model, and train and test your model
    for one epoch. The number of training steps should be your the number of
    batches you run through in a single epoch. You should receive a final accuracy on the testing examples of > 80%.
    :return: None
    """
    trainingQuantityModel = 60000
    testQuantity = 10000    
    # TODO: load MNIST train and test examples into train_inputs, train_labels, test_inputs, test_labels

    trainImgs, trainLabels = get_data(f'{mnist_data_folder}/train-images-idx3-ubyte.gz', f'{mnist_data_folder}/train-labels-idx1-ubyte.gz', trainingQuantityModel)

    testImgs, testLabels = get_data(f'{mnist_data_folder}/t10k-images-idx3-ubyte.gz', f'{mnist_data_folder}/t10k-labels-idx1-ubyte.gz', testQuantity)
    # TODO: Create Model
    model = Model()
    # TODO: Train model by calling train() ONCE on all data
    train(model, trainImgs, trainLabels)
    # TODO: Test the accuracy by calling test() after running train()
    #test()?
    test(model, testImgs,testLabels)
    # TODO: Visualize the data by using visualize_results()
    
    # for interest
    #test(model, trainImgs,trainLabels)
    snip_size = 20
    start_point_for_vs = 10
    visualize_results(testImgs[start_point_for_vs:snip_size], model.call(testImgs[start_point_for_vs:snip_size]), testLabels[start_point_for_vs:snip_size])

    print("end of assignment 1")    



if __name__ == '__main__':
    #TODO: you might need to change this to something else if you run locally
    main("./MNIST_data")
    
