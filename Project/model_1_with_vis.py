import os
# suppress silly log messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



test_mses = {}
val_mses = {}
train_mses = {}

import matplotlib.pyplot as plt
from tensorflow.io import FixedLenFeature, FixedLenSequenceFeature

import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend first




import tensorflow as tf
import numpy as np
import random
import structure_prediction_utils as utils
from tensorflow import keras
import sys
class ProteinStructurePredictor0(keras.Model):

    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3 = keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.batch_norm = keras.layers.BatchNormalization()
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(256 * 3, activation='relu')
        self.final_dense = keras.layers.Dense(256 * 3, activation='linear')
        
        self.layer1 = keras.layers.Conv2D(1, 1, activation='gelu', padding="same")
        self.normalize = keras.layers.LayerNormalization()
        

    def compute_euclidean_distances(self, points):
        expanded_points = tf.expand_dims(points, axis=1)
        expanded_points_transpose = tf.transpose(expanded_points, perm=[0, 2, 1, 3])
   
        squared_differences = tf.square(expanded_points - expanded_points_transpose)
        squared_distances = tf.reduce_sum(squared_differences, axis=-1)
        
        return squared_distances



    #@tf.function
    def call(self, inputs, mask=None):
        primary_one_hot = inputs['primary_onehot']
        primary_one_hot = tf.expand_dims(primary_one_hot, -1)
        evo = inputs['evo']
        this_batch_size = tf.shape(evo)[0]
        evo = tf.reshape(evo, (this_batch_size, 256, 21))
        evo = tf.expand_dims(evo, -1)
        together = tf.concat([evo, primary_one_hot], axis=-1)
        x = self.conv1(together)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.batch_norm(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = tf.reshape(x, (tf.shape(x)[0], 256, 3))
        distance_matrix = self.compute_euclidean_distances(x[:, :, :])
        return distance_matrix




def get_n_records(batch):
    return batch['primary_onehot'].shape[0]
def get_input_output_masks(batch):
    inputs = {'primary_onehot':batch['primary_onehot'], 'primary':batch['primary'], 'evo':batch['evolutionary']}
    outputs = batch['true_distances']
    masks = batch['distance_mask']

    return inputs, outputs, masks


def train(epochnum, model, train_dataset, validate_dataset=None, train_loss=utils.mse_loss):
    '''
    Trains the model
    '''

    avg_loss = 0.
    avg_mse_loss = 0.

    def print_loss():
        if validate_dataset is not None:
            validate_loss = 0.

            validate_batches = 0.
            for batch in validate_dataset.batch(1024):
                validate_inputs, validate_outputs, validate_masks = get_input_output_masks(batch)
                validate_preds = model.call(validate_inputs, validate_masks)

                validate_loss += tf.reduce_sum(utils.mse_loss(validate_preds, validate_outputs, validate_masks)) / get_n_records(batch)
                validate_batches += 1
            validate_loss /= validate_batches
        else:
            validate_loss = float('NaN')
       # print(
        #    f'train loss {avg_loss:.3f} train mse loss {avg_mse_loss:.3f} validate mse loss {validate_loss:.3f}')
    
    
    
        
        val_mses[epochnum].append(validate_loss.numpy())
        train_mses[epochnum].append(avg_mse_loss.numpy())
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    first = True
    for batch in train_dataset:
        inputs, labels, masks = get_input_output_masks(batch)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = model(inputs, masks)
            l = train_loss(outputs, labels, masks)
            batch_loss = tf.reduce_sum(l)
            gradients = tape.gradient(batch_loss, model.trainable_weights)
            avg_loss = batch_loss / get_n_records(batch)
            avg_mse_loss = tf.reduce_sum(utils.mse_loss(outputs, labels, masks)) / get_n_records(batch)

        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        print_loss()

        if first:
            print(model.summary())
            first = False

def test(model, epochnum, test_records, viz=False):
    for batch in test_records.batch(1024):
        test_inputs, test_outputs, test_masks = get_input_output_masks(batch)
        test_preds = model.call(test_inputs, test_masks)
        test_loss = tf.reduce_sum(utils.mse_loss(test_preds, test_outputs, test_masks)) / get_n_records(batch)
        print(f'test mse loss {test_loss:.3f}')
        test_mses[epochnum].append(test_loss.numpy())

    if viz:
       # print(model.summary())
        r = random.randint(0, test_preds.shape[0])
        utils.display_two_structures(test_preds[r], test_outputs[r], test_masks[r])

def main(data_folder):
    training_records = utils.load_preprocessed_data(data_folder, 'training.tfr')
    validate_records = utils.load_preprocessed_data(data_folder, 'validation.tfr')
    test_records = utils.load_preprocessed_data(data_folder, 'testing.tfr')

    model = ProteinStructurePredictor0()
    model.optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.batch_size = 512
    epochs = 25
    
    
    
    
    
    
    
    
    
    
    
    for key in range(epochs):
        test_mses[key] = []
        val_mses[key] = []
        train_mses[key] = []
    test_mses[-1] = []






    # Iterate over epochs.
    for epoch in range(epochs):
        epoch_training_records = training_records.shuffle(buffer_size=256).batch(model.batch_size, drop_remainder=False)
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        train(epoch,model, epoch_training_records, validate_records)

        test(model, epoch, test_records, False)

    test(model, -1, test_records, True)
    print(test_mses)
    print(val_mses)
    print(train_mses)
    print("----")
    final_test_ave = test_mses[-1]
            
    del(test_mses[-1])
    print("FINAL TeS MSE", final_test_ave)    
        
    print("test_mSes at each epoch", avgs(test_mses))
    print("\n")
    print("train_MSes at each epoch", avgs(train_mses))
    print("\n")
    print("val at each epoch", avgs(val_mses))
    
    

    plot_mse(avgs(train_mses),avgs(val_mses),avgs(test_mses))
    
    
    
    


    
    
    model.save(data_folder + '/model')


def avgs(input_dict):
    averages = []
    
    for key in sorted(input_dict.keys()):  # Sort the keys to maintain order
        value_list = input_dict[key]
        if value_list:  # Check if the list is not empty
            averages.append(sum(value_list) / len(value_list))
        else:
            averages.append(0)  # Or handle empty lists as needed

    return averages
def plot_mse(train_mse, validate_mse, test_mse):
    """
    Plots the training, validation, and test mean squared errors (MSE) over epochs.

    Parameters:
    - train_mse: List of training MSE values.
    - validate_mse: List of validation MSE values.
    - test_mse: List of test MSE values.
    """
    epochs = range(1, len(train_mse) + 1)  # Create a range for the number of epochs

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_mse, label='Training MSE', marker='o')
    plt.plot(epochs, validate_mse, label='Validation MSE', marker='o')
    plt.plot(epochs, test_mse, label='Test MSE', marker='o')
    
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == '__main__':
    local_home = os.path.expanduser("~")  # on my system this is /Users/jat171
    data_folder = local_home 
    print(data_folder)
    
    # local 
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_folder = os.path.join(current_dir, 'data/')
    print(data_folder)





    main(data_folder)