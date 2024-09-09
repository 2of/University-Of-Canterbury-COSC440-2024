from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
    """
    Performs 2D convolution given 4D inputs and filter Tensors.
    :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
    :param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
    :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
    :param padding: either "SAME" or "VALID", capitalization matters
    :return: outputs, NumPy array or Tensor with shape [num_examples, output_height, output_width, output_channels]
    """
    print("PADDING:", padding)
    
    # Input dimensions
    num_examples = inputs.shape[0]
    in_height = inputs.shape[1]
    in_width = inputs.shape[2]
    input_in_channels = inputs.shape[3]

    # Filter dimensions
    filter_height = filters.shape[0]
    filter_width = filters.shape[1]
    filter_in_channels = filters.shape[2]
    filter_out_channels = filters.shape[3]

    # Strides
    strideY = strides[1]
    strideX = strides[2]

    # Padding calculation
    if padding == "SAME":
        pad_h = max((in_height - 1) * strideY + filter_height - in_height, 0)
        pad_w = max((in_width - 1) * strideX + filter_width - in_width, 0)
        # stride is always 1 yet adding * strideY and * strideX I think is what	
        #passed the tests
        
        t = pad_h // 2
        b = pad_h - t
        l = pad_w // 2
        r = pad_w - l
        inputs = np.pad(inputs, [[0, 0], [t, b], [l, r], [0, 0]], mode='constant')

        out_height = math.ceil(in_height / strideY)
        out_width = math.ceil(in_width / strideX)

    elif padding == "VALID":
        out_height = (in_height - filter_height) // strideY + 1
        out_width = (in_width - filter_width) // strideX + 1

    # Perform convolution
    outputs = np.zeros((num_examples, out_height, out_width, filter_out_channels))
    
    for i in range(out_height):
        for j in range(out_width):
            for k in range(filter_out_channels):
                v1 = i * strideY
                v2 = v1 + filter_height
                h1 = j * strideX
                h2 = h1 + filter_width
                
                image_chunk = inputs[:, v1:v2, h1:h2, :]
                filter_slice = filters[:, :, :, k]
                conv_sum = np.sum(image_chunk * filter_slice, axis=(1, 2, 3))
                outputs[:, i, j, k] = conv_sum

    return outputs