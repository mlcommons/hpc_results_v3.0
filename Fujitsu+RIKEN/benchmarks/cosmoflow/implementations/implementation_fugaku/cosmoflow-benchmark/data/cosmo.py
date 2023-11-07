"""CosmoFlow dataset specification"""

# System imports
import os
import logging
import glob
from functools import partial

# External imports
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf

from mlperf_logging import mllog
#import horovod.tensorflow.keras as hvd

# Local imports
from utils.staging import stage_files

def _augment_data(x, y, seed):
    # random reverse
    rev_idx = tf.random.uniform([3], minval=0, maxval=2, dtype=tf.int32, seed=seed)
    rev_idx = tf.where(rev_idx > 0)
    rev_idx = tf.reshape(rev_idx, [-1])
    # rev_idx = tf.Print(rev_idx, [rev_idx], message="[{}] Reverse idx: ".format(dist.rank))
    x = tf.reverse(x, rev_idx)

    # random transpose
    random_space_idx = tf.random.shuffle(tf.constant([0,1,2]), seed=seed)
    # random_space_idx = tf.Print(random_space_idx, [random_space_idx], message="[{}] Transpose idx: ".format(dist.rank))
    random_idx = tf.concat([random_space_idx, tf.constant([3])], 0)
    x = tf.transpose(x, perm=random_idx)
    return x, y

def _parse_data(sample_proto, shape):
    """Parse the data out of the TFRecord proto buf.

    This pipeline could be sped up considerably by moving the cast and log
    transform onto the GPU, in the model (e.g. in a keras Lambda layer).
    """

    # Parse the serialized features
    feature_spec = dict(x=tf.io.FixedLenFeature([], tf.string),
                        y=tf.io.FixedLenFeature([4], tf.float32))
    parsed_example = tf.io.parse_single_example(
        sample_proto, features=feature_spec)

    # Decode the bytes data
    x = tf.reshape(tf.io.decode_raw(parsed_example['x'], tf.int16), shape)
    y = parsed_example['y']

    return x, y

def _preproc_data(x, y, shape, apply_log):
    # convert to float
    x = tf.cast(x, tf.float32)
    
    # Data normalization/scaling
    if apply_log:
        # Take logarithm of the data spectrum
        x = tf.math.log(x + tf.constant(1.))
        #x = tf.math.log1p(x)
    else:
        # Traditional mean normalization
        x /= (tf.reduce_sum(x) / np.prod(shape))
    return x, y

def _slice_data(x, y, shape, dist):
    # get model parallel slice
    #tf.logging.info("[{}] input model parallel seed={}".format(dist.rank, seed))
    dep_size = shape[0] // dist.model_parallel_size[0]
    row_size = shape[1] // dist.model_parallel_size[1]
    x = x[dep_size*(dist.model_parallel_rank[0]):dep_size*(dist.model_parallel_rank[0]+1),
          row_size*(dist.model_parallel_rank[1]):row_size*(dist.model_parallel_rank[1]+1),
          :,:]
    return x, y

def construct_dataset(file_dir, n_samples, batch_size, n_epochs,
                      sample_shape, samples_per_file=1, n_file_sets=1,
                      shard=0, n_shards=1, apply_log=False, compression=None,
                      randomize_files=False, shuffle=False,
                      shuffle_buffer_size=0, prefetch=4, seed=None, do_augmentation=False, dist=None,
                      use_cache=False, cache_as_fp32=False):
    """This function takes a folder with files and builds the TF dataset.

    It ensures that the requested sample counts are divisible by files,
    local-disks, worker shards, and mini-batches.
    """

    if n_samples == 0:
        return None, 0

    # Ensure samples divide evenly into files * local-disks * worker-shards * batches
    n_divs = samples_per_file * n_file_sets * n_shards * batch_size
    if (n_samples % n_divs) != 0:
        logging.error('Number of samples (%i) not divisible by %i '
                      'samples_per_file * n_file_sets * n_shards * batch_size',
                      n_samples, n_divs)
        raise Exception('Invalid sample counts')

    # Number of files and steps
    n_files = n_samples // (samples_per_file * n_file_sets)
    n_steps = n_samples // (n_file_sets * n_shards * batch_size)

    # Find the files
    filenames = sorted(glob.glob(os.path.join(file_dir, '*.tfrecord')))
    assert (0 <= n_files) and (n_files <= len(filenames)), (
        'Requested %i files, %i available' % (n_files, len(filenames)))
    if randomize_files:
        np.random.shuffle(filenames)
    filenames = filenames[:n_files]

    def data_fn():
        # Define the dataset from the list of sharded, shuffled files
        data = tf.data.Dataset.from_tensor_slices(filenames)
        data = data.shard(num_shards=n_shards, index=shard)

        shape = sample_shape

        if (not use_cache) and shuffle:
            data = data.shuffle(len(filenames), reshuffle_each_iteration=True, seed=seed)

        # Parse TFRecords
        parse_data = partial(_parse_data, shape=shape)
        def dataset_fn(ds):
            return tf.data.TFRecordDataset(ds, num_parallel_reads=1, compression_type=compression)
        data = data.apply(dataset_fn).map(parse_data, num_parallel_calls=1)

        if cache_as_fp32:
            preproc_data = partial(_preproc_data, shape=shape, apply_log=apply_log)
            data = data.map(preproc_data)

        if use_cache:
            data = data.cache()
            if shuffle:
                data = data.shuffle(len(filenames), reshuffle_each_iteration=True, seed=seed)

        if do_augmentation:
            augment_data = partial(_augment_data, seed=seed)
            data = data.map(augment_data)

        if dist:
            slice_data = partial(_slice_data, shape=shape, dist=dist)
            data = data.map(slice_data)
            shape = (shape[0] // dist.model_parallel_size[0],
                     shape[1] // dist.model_parallel_size[1],
                     shape[2],
                     shape[3])

        if not cache_as_fp32:
            preproc_data = partial(_preproc_data, shape=shape, apply_log=apply_log)
            data = data.map(preproc_data)

        # Parallelize reading with interleave - no benefit?
        #data = data.interleave(
        #    lambda x: tf.data.TFRecordDataset(x).map(parse_data, num_parallel_calls=1),
        #    cycle_length=4
        #)

        # Localized sample shuffling (note: imperfect global shuffling).
        # Use if samples_per_file is greater than 1.
        if shuffle and shuffle_buffer_size > 0:
            data = data.shuffle(shuffle_buffer_size)

        # Construct batches
        data = data.repeat(n_epochs)
        data = data.batch(batch_size, drop_remainder=True)

        # Prefetch to device
        data = data.prefetch(prefetch)
        return data


    return data_fn, n_steps


def get_datasets(data_dir, sample_shape, n_train, n_valid,
                 batch_size, n_epochs, dist, samples_per_file=1,
                 shuffle_train=True, shuffle_valid=False,
                 shard=True, stage_dir=None,
                 prefetch=4, apply_log=False, prestaged=False, seed=-1, do_augmentation=False, validation_batch_size=None, train_staging_dup_factor=1, use_cache=False, cache_as_fp32=False, compression=None):
    """Prepare TF datasets for training and validation.

    This function will perform optional staging of data chunks to local
    filesystems. It also figures out how to split files according to local
    filesystems (if pre-staging) and worker shards (if sharding).

    Returns: A dict of the two datasets and step counts per epoch.
    """

    # MLPerf logging
    # mllogger = mllog.get_mllogger()
    # mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=batch_size*dist.size)
    # mllogger.event(key=mllog.constants.TRAIN_SAMPLES, value=n_train)
    # mllogger.event(key=mllog.constants.EVAL_SAMPLES, value=n_valid)
    data_dir = os.path.expandvars(data_dir)

    # Local data staging
    if stage_dir is not None:
        staged_files = True
        if prestaged:
            if (dist.rank == 0):
                print('data is alreadly staged')
        else:
            if (dist.rank == 0):
                print('data is not staged yet')
            # Stage training data
            stage_files(os.path.join(data_dir, 'train'),
                        os.path.join(stage_dir, 'train'),
                        n_files=n_train, rank=dist.rank, size=dist.size)
            # Stage validation data
            stage_files(os.path.join(data_dir, 'validation'),
                        os.path.join(stage_dir, 'validation'),
                        n_files=n_valid, rank=dist.rank, size=dist.size)

            # Barrier to ensure all workers are done transferring
            if dist.size > 0:
                dist.barrier()
        data_dir = stage_dir
    else:
        staged_files = False

    # Determine number of staged file sets and worker shards
    if (dist.data_parallel_size // dist.data_parallel_local_size) % train_staging_dup_factor != 0:
        raise Exception('# nodes is not a multiple of train_staging_dup_factor')

    n_train_file_sets = (dist.data_parallel_size // (dist.data_parallel_local_size * train_staging_dup_factor)) if staged_files else 1
    if shard and staged_files:
        n_train_shards = dist.data_parallel_local_size * train_staging_dup_factor
        train_shard = dist.data_parallel_rank % n_train_shards
    elif shard and not staged_files:
        train_shard, n_train_shards = dist.data_parallel_rank, dist.data_parallel_size
    else:
        train_shard, n_train_shards = 0, 1

    n_valid_file_sets = (dist.data_parallel_size // dist.data_parallel_local_size) if staged_files else 1
    if shard and staged_files:
        valid_shard, n_valid_shards = dist.data_parallel_local_rank, dist.data_parallel_local_size
    elif shard and not staged_files:
        valid_shard, n_valid_shards = dist.data_parallel_rank, dist.data_parallel_size
    else:
        valid_shard, n_valid_shards = 0, 1

    rank_seed = seed + dist.data_parallel_rank if seed >= 0 else dist.data_parallel_rank # use same seed in model parallel
    #FIXME for seed < -1, add base seed broadcasted from root

    # Construct the training and validation datasets
    train_dataset_args = dict(batch_size=batch_size, n_epochs=n_epochs,
                              sample_shape=sample_shape, samples_per_file=samples_per_file,
                              n_file_sets=n_train_file_sets, shard=train_shard, n_shards=n_train_shards,
                              apply_log=apply_log, prefetch=prefetch, dist=dist, use_cache=use_cache, cache_as_fp32=cache_as_fp32, compression=compression)
    train_dataset, n_train_steps = construct_dataset(
        file_dir=os.path.join(data_dir, 'train'),
        n_samples=n_train, shuffle=shuffle_train, seed=rank_seed, do_augmentation=do_augmentation, **train_dataset_args)

    valid_dataset_args = dict(batch_size=validation_batch_size or batch_size, n_epochs=n_epochs,
                              sample_shape=sample_shape, samples_per_file=samples_per_file,
                              n_file_sets=n_valid_file_sets, shard=valid_shard, n_shards=n_valid_shards,
                              apply_log=apply_log, prefetch=prefetch, dist=dist, use_cache=use_cache, cache_as_fp32=cache_as_fp32, compression=compression)
    valid_dataset, n_valid_steps = construct_dataset(
        file_dir=os.path.join(data_dir, 'validation'),
        n_samples=n_valid, shuffle=shuffle_valid, **valid_dataset_args)

    if shard == 0:
        if staged_files:
            logging.info('Using %i locally-staged train file sets and %i locally-staged validation file sets', n_train_file_sets, n_valid_file_sets)
        logging.info('Splitting data into %i train worker shards and validation worker shards', n_train_shards, n_valid_shards)
        n_train_worker = n_train // (samples_per_file * n_train_file_sets * n_train_shards)
        n_valid_worker = n_valid // (samples_per_file * n_valid_file_sets * n_valid_shards)
        logging.info('Each worker reading %i training samples and %i validation samples',
                     n_train_worker, n_valid_worker)

    return dict(train_dataset=train_dataset, valid_dataset=valid_dataset,
                n_train_steps=n_train_steps, n_valid_steps=n_valid_steps)
