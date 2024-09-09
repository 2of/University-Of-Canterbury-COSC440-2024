import gzip
IMAGE_BUFF_SIZE = 16
LABELS_BUFF_SIZE = 8
IMAGE_SIZE_W = 28
IMAGE_SIZE_H = 28
import numpy as np 


def get_data(inputs_file_path, labels_file_path, num_examples):
    """
    Takes in an inputs file path and labels file path, unzips both files,
    normalizes the inputs, and returns (NumPy array of inputs, NumPy
    array of labels). Read the data of the file into a buffer and use
    np.frombuffer to turn the data into a NumPy array. Keep in mind that
    each file has a header of a certain size. This method should be called
    within the main function of the assignment.py file to get BOTH the train and
    test data. If you change this method and/or write up separate methods for
    both train and test data, we will deduct points.

    Hint: look at the writeup for sample code on using the gzip library

    :param inputs_file_path: file path for inputs, something like
    'MNIST_data/t10k-images-idx3-ubyte.gz'
    :param labels_file_path: file path for labels, something like
    'MNIST_data/t10k-labels-idx1-ubyte.gz'
    :param num_examples: used to read from the bytestream into a buffer. Rather
    than hardcoding a number to read from the bytestream, keep in mind that each image
    (example) is 28 * 28, with a header of a certain number.
    :return: NumPy array of inputs as float32 and labels as int8
    """

    # TODO: Load inputs and labels
    
    
    def gzip_read_to_buffer(file_path, header_size, num_elements):
        with gzip.open(file_path, 'rb') as f:
            f.read(header_size)  # Ignore header, size specified at top of page
            buffer = f.read(num_elements)
        return buffer

    # Read the input images
    input_buffer = gzip_read_to_buffer(inputs_file_path, IMAGE_BUFF_SIZE, num_examples * IMAGE_SIZE_W * IMAGE_SIZE_H)
    inputs = np.frombuffer(input_buffer, dtype=np.uint8).astype(np.float32)
    inputs = inputs.reshape(num_examples, IMAGE_SIZE_W * IMAGE_SIZE_H)

    inputs /= 255  # Normalize by dividing by 255


    label_buffer = gzip_read_to_buffer(labels_file_path, LABELS_BUFF_SIZE, num_examples)
    labels = np.frombuffer(label_buffer, dtype=np.uint8).astype(np.int8)

    return inputs, labels
