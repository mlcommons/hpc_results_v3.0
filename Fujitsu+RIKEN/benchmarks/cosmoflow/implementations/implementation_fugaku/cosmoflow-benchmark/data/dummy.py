"""
Random dummy dataset specification.
"""

# System
import math
from functools import partial

# Externals
import numpy as np
import tensorflow.compat.v1 as tf
from .cosmo import _augment_data

def _parse_data(x, y, shape, apply_log=False, seed=None, do_augmentation=False, dist=None):
    if do_augmentation:
        _augment_data(x, seed)

    # get model parallel slice
    if dist:
        dep_size = shape[0] // dist.model_parallel_size[0]
        row_size = shape[1] // dist.model_parallel_size[1]
        x = x[dep_size*(dist.model_parallel_rank[0]):dep_size*(dist.model_parallel_rank[0]+1),
              row_size*(dist.model_parallel_rank[1]):row_size*(dist.model_parallel_rank[1]+1),
              :,:]
        shape = [dep_size, row_size] + shape[2:]

    # convert to float
    x = tf.cast(x, tf.float32)

    # Data normalization/scaling
    if apply_log:
        # Take logarithm of the data spectrum
        x = tf.math.log(x + tf.constant(1.))
    else:
        # Traditional mean normalization
        x /= (tf.reduce_sum(x) / np.prod(shape))

    return x, y

def construct_dataset(sample_shape, target_shape,
                      batch_size=1, n_samples=32,
                      prefetch=4, apply_log=False, seed=None, do_augmentation=False,
                      dist=None):
    def data_fn():
        x = tf.random.uniform([n_samples]+sample_shape, maxval=255, dtype=tf.int32) # int16 is not supported
        y = tf.random.uniform([n_samples]+target_shape, minval=-1, maxval=1, dtype=tf.float32)
        data = tf.data.Dataset.from_tensor_slices((x, y))

        parse_data = partial(_parse_data, shape=sample_shape,
                             apply_log=apply_log, seed=seed, do_augmentation=do_augmentation, dist=dist)

        data = data.map(parse_data, num_parallel_calls=4)
        data = data.repeat().batch(batch_size)

        data = data.prefetch(prefetch)

        return data

    return data_fn

def get_datasets(sample_shape, target_shape, batch_size,
                 n_train, n_valid, dist, n_epochs=None, shard=False, seed=-1,
                 prefetch=4, apply_log=False, do_augmentation=False):
    train_dataset = construct_dataset(sample_shape, target_shape, batch_size=batch_size,
                                      prefetch=prefetch, apply_log=apply_log,
                                      do_augmentation=do_augmentation, dist=dist)
    valid_dataset = None
    if n_valid > 0:
        valid_dataset = construct_dataset(sample_shape, target_shape, batch_size=batch_size,
                                          prefetch=prefetch, apply_log=apply_log, dist=dist)
    n_train_steps = n_train  // batch_size
    n_valid_steps = n_valid  // batch_size
    if shard:
        n_train_steps = n_train_steps // dist.data_parallel_size
        n_valid_steps = n_valid_steps // dist.data_parallel_size

    return dict(train_dataset=train_dataset, valid_dataset=valid_dataset,
                n_train=n_train, n_valid=n_valid, n_train_steps=n_train_steps,
                n_valid_steps=n_valid_steps)
