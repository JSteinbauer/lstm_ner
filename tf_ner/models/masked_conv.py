"""Implement masked 1d convolution with max-pooling"""

__author__ = "Guillaume Genthial"

import tensorflow as tf
from six.moves import reduce
from tensorflow import Tensor


def masked_conv1d_and_max(t: Tensor, weights: Tensor, filters: int, kernel_size: int) -> Tensor:
    """Applies 1d convolution and a masked max-pooling

    Parameters
    ----------
    t : tf.Tensor
        A tensor with at least 3 dimensions [d1, d2, ..., dn-1, dn]
    weights : tf.Tensor of tf.bool
        A Tensor of shape [d1, d2, dn-1]
    filters : int
        number of filters
    kernel_size : int
        kernel size for the temporal convolution

    Returns
    -------
    tf.Tensor
        A tensor of shape [d1, d2, dn-1, filters]

    """
    # Get shape and parameters
    shape = tf.shape(t)
    ndims = t.shape.ndims
    dim1 = reduce(lambda x, y: x * y, [shape[i] for i in range(ndims - 2)])
    dim2 = tf.reshape(tf.slice(shape, [2], [1]), [])
    dim3 = t.shape[-1]

    # Reshape weights
    weights = tf.reshape(weights, shape=[dim1, dim2, 1])
    weights = tf.cast(weights, tf.float32)

    # Reshape input and apply weights
    flat_shape = [dim1, dim2, dim3]
    t = tf.reshape(t, shape=flat_shape)
    t *= weights

    # Apply convolution
    t_conv = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(
        t)  # tf.layers.conv1d(t, filters, kernel_size, padding='same')
    t_conv *= weights

    # Reduce max -- set to zero if all padded
    t_conv += (1. - weights) * tf.reduce_min(t_conv, axis=-2, keepdims=True)
    t_max = tf.reduce_max(t_conv, axis=-2)

    # Reshape the output
    final_shape = [shape[i] for i in range(ndims - 2)] + [filters]
    t_max = tf.reshape(t_max, shape=final_shape)

    return t_max
