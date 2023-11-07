# coding=utf-8
# Copyright 2020 The Mesh TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MNIST using Mesh TensorFlow and TF Estimator.

This is an illustration, not a good model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

import glob
import os
from functools import partial

import numpy as np
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf

from mesh_tensorflow import logging_util as logger

mesh_width =1 
mesh_height = 1
num_active_proc = 1
mtf.mesh_init(axis_d_block=mesh_height, axis_h_block=mesh_width, num_active_proc=num_active_proc)

num_model_procs = mesh_height * mesh_width
meshshape="b1:"+str(1)+";b2:"+str(1)
meshdev = [""]
#meshdev = ["gpu:0"]
#meshdev = ["gpu:0","gpu:1"]
#meshdev = ["gpu:0","gpu:1","gpu:2","gpu:3"]
#meshdev = ["gpu:0","gpu:1","gpu:2","gpu:3",
#        "gpu:4","gpu:5","gpu:6","gpu:7"]
#meshdev = ["gpu:0","gpu:0","gpu:1","gpu:1",
#        "gpu:2","gpu:2","gpu:3","gpu:3",
#        "gpu:4","gpu:4","gpu:5","gpu:5",
#        "gpu:6","gpu:6","gpu:7","gpu:7"]

tf.flags.DEFINE_string("data_dir", "/home/kasagi/COSMOFLOW/cosmoflow-benchmark/data/tfrecord",
                       "Path to directory containing the cosmoflow dataset")
tf.flags.DEFINE_string("model_dir", "/home/kasagi/COSMOFLOW/cosmoflow_model", "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 1,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_float("dropout", 0.5, "Dropout Rate.")
tf.flags.DEFINE_integer("train_epochs", 1, "Total number of training epochs.")
tf.flags.DEFINE_integer("epochs_between_evals", 1,
                        "# of epochs between evaluations.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_string("mesh_shape", meshshape, "mesh shape")
tf.flags.DEFINE_string("layout", "dep_blocks:b1;row_blocks:b2",
                       "layout rules")

FLAGS = tf.flags.FLAGS


def cosmo_model(image, labels, mesh):
  """The model.

  Args:
    image: tf.Tensor with shape [batch, 28*28]
    labels: a tf.Tensor with shape [batch] and dtype tf.int32
    mesh: a mtf.Mesh

  Returns:
    logits: a mtf.Tensor with shape [batch, 10]
    loss: a mtf.Tensor with shape []
  """
  batch_dim = mtf.Dimension("batch", FLAGS.batch_size)
  dep_blocks_dim = mtf.Dimension("dep_blocks", 1)
  row_blocks_dim = mtf.Dimension("row_blocks", 1)
  col_blocks_dim = mtf.Dimension("col_blocks", 1)
  deps_dim = mtf.Dimension("deps_size", 128 // mesh_height)
  rows_dim = mtf.Dimension("rows_size", 128 // mesh_width)
  cols_dim = mtf.Dimension("cols_size", 128)

  channel_dim = mtf.Dimension("one_channel", 4)

  x = mtf.import_tf_tensor(
      #mesh, tf.reshape(image, [FLAGS.batch_size, 4, 32, 4, 32, 1, 128, 4]),
      mesh, tf.reshape(image, [FLAGS.batch_size, 1, 128//mesh_height,
          1, 128//mesh_width, 1, 128, 4]),
      mtf.Shape(
          [batch_dim, dep_blocks_dim, deps_dim, row_blocks_dim, rows_dim,
           col_blocks_dim, cols_dim, channel_dim]))
  x = mtf.transpose(x, [
      batch_dim, dep_blocks_dim, row_blocks_dim, col_blocks_dim,
      deps_dim, rows_dim, cols_dim, channel_dim])

  # add some convolutional layers to demonstrate that convolution works.
  _num_filters=5
  _filter=[3, 3, 3]
  filters_dim=[]*_num_filters
  for i in range(_num_filters):
    filters_dim.append(mtf.Dimension("filters"+str(i + 1), 32 * (2 ** i)))
    x = mtf.layers.conv3d_with_MPI(
        x, filters_dim[i], filter_size=_filter, strides=[1, 1, 1], padding="SAME",
        d_blocks_dim=dep_blocks_dim, h_blocks_dim=row_blocks_dim,
        name="conv"+str(i))
    bias = mtf.get_variable(
           x.mesh,
           "conv_bias"+str(i),
           mtf.Shape([filters_dim[i]]),
           initializer=tf.zeros_initializer(),
           dtype=tf.float32)
    x += bias
    x = mtf.layers.max_pool3d(mtf.relu(x))

  # add some fully-connected dense layers.
  hidden_dim1 = mtf.Dimension("hidden1", 128)
  hidden_dim2 = mtf.Dimension("hidden2", 64)
  hidden_dim3 = mtf.Dimension("hidden3", 4)
  x = mtf.layers.conv3d_to_dense(x, mesh_size=[mesh_height,mesh_width])
  h1 = mtf.layers.dense(x, hidden_dim1, reduced_dims=x.shape.dims[-7:],
          activation=mtf.leaky_relu, name="hidden1_l2")
  d1 = mtf.dropout(h1, 1.0 - FLAGS.dropout)
  h2 = mtf.layers.dense(d1, hidden_dim2, activation=mtf.leaky_relu,
          name="hidden2_l2")
  d2 = mtf.dropout(h2, 1.0 - FLAGS.dropout)
  h3 = mtf.layers.dense(d2, hidden_dim3, activation=mtf.leaky_relu,
          name="hidden3")
  logits = mtf.multiply(h3, 1.2)
  loss = mtf.square(logits - tf.reshape(labels, (FLAGS.batch_size,4)))
  loss = mtf.reduce_mean(loss)
  return logits, loss


def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  logger.info("features = %s labels = %s mode = %s params=%s" %
                  (features, labels, mode, params))

  global_step = tf.train.get_global_step()
  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "my_mesh")
  logits, loss = cosmo_model(features, labels, mesh)
  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)
  mesh_size = mesh_shape.size
  mesh_devices = meshdev
  mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
      mesh_shape, layout_rules, mesh_devices)

  if mode == tf.estimator.ModeKeys.TRAIN:
    var_grads = mtf.gradients(
        [loss], [v.outputs[0] for v in graph.trainable_variables])
    optimizer = mtf.optimize.MomentumOptimizer(learning_rate=0.0001, momentum=0.9)
    update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

  lowering = mtf.Lowering(graph, {mesh: mesh_impl})
  restore_hook = mtf.MtfRestoreHook(lowering)

  tf_logits = lowering.export_to_tf_tensor(logits)
  if mode != tf.estimator.ModeKeys.PREDICT:
    tf_loss = lowering.export_to_tf_tensor(loss)
    tf.summary.scalar("loss", tf_loss)

  if mode == tf.estimator.ModeKeys.TRAIN:
    tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
    tf_update_ops.append(tf.assign_add(global_step, 1))
    train_op = tf.group(tf_update_ops)
#    if mtf.get_model_rank() == 0:
    if comm_rank == 0:
       saver = tf.train.Saver(
           tf.global_variables(),
           sharded=True,
           max_to_keep=10,
           keep_checkpoint_every_n_hours=2,
           defer_build=False, save_relative_paths=True)
       tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
       saver_listener = mtf.MtfCheckpointSaverListener(lowering)
       saver_hook = tf.train.CheckpointSaverHook(
           FLAGS.model_dir,
           save_steps=1000,
           saver=saver,
           listeners=[saver_listener])

    mae = tf.metrics.mean_absolute_error(
        labels=labels, predictions=tf_logits)

    # Name tensors to be logged with LoggingTensorHook.
    tf.identity(tf_loss, "cross_entropy")
    tf.identity(mae[1], name="train_mae")

    # Save accuracy scalar to Tensorboard output.
    tf.summary.scalar("train_mae", mae[1])

    # restore_hook must come before saver_hook
    if comm_rank == 0:
        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.TRAIN, loss=tf_loss, train_op=train_op,
            training_chief_hooks=[restore_hook])
            #training_chief_hooks=[restore_hook, saver_hook])
    else:
        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.TRAIN, loss=tf_loss, train_op=train_op,
            training_chief_hooks=[restore_hook])

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        "classes": tf.argmax(tf_logits, axis=1),
        "probabilities": tf.nn.softmax(tf_logits),
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        prediction_hooks=[restore_hook],
        export_outputs={
            "classify": tf.estimator.export.PredictOutput(predictions)
        })
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=tf_loss,
        evaluation_hooks=[restore_hook],
        eval_metric_ops={
            "mae":
            tf.metrics.mean_absolute_error(
                labels=labels, predictions=tf_logits),
        })


def run_cosmo():
  """Run MNIST training and eval loop."""
  session_config = tf.compat.v1.ConfigProto(
                inter_op_parallelism_threads=1)
  run_config=tf.estimator.RunConfig(session_config=session_config)
  cosmo_classifier = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=FLAGS.model_dir,
      config=run_config)

  # Set up training and evaluation input functions.
  def train_input_fn_dmy():
    """Prepare data for training."""

    x = tf.random.uniform([10, 128//mesh_height, 128//mesh_width, 128, 4])
    y = tf.random.uniform([10, 4])
    data = tf.data.Dataset.from_tensor_slices((x,y))
    return data.repeat().batch(FLAGS.batch_size).prefetch(1)

  def _parse_data(sample_proto, shape, apply_log=False):
    """Parse the data out of the TFRecord proto buf.

    This pipeline could be sped up considerably by moving the cast and log
    transform onto the GPU, in the model (e.g. in a keras Lambda layer).
    """

    # Parse the serialized features
    feature_spec = dict(x=tf.io.FixedLenFeature([], tf.string),
                        y=tf.io.FixedLenFeature([4], tf.float32))
    parsed_example = tf.io.parse_single_example(
        sample_proto, features=feature_spec)

    # Decode the bytes data, convert to float
    x = tf.io.decode_raw(parsed_example['x'], tf.int16)
    x = tf.cast(tf.reshape(x, shape), tf.float32)
    y = parsed_example['y']

    # Data normalization/scaling
    if apply_log:
        # Take logarithm of the data spectrum
        x = tf.math.log(x + tf.constant(1.))
    else:
        # Traditional mean normalization
        x /= (tf.reduce_sum(x) / np.prod(shape))

    return x, y

  def _data_input_fn(name):
    """Prepare data for training."""
    file_dir=os.path.join(FLAGS.data_dir, name)
    n_samples=128
    batch_size=FLAGS.batch_size
    n_epochs=2
    sample_shape=[128, 128, 128, 4]
    samples_per_file=1
    n_file_sets=1
    shard=0
    n_shards=1
    apply_log=False
    randomize_files=False
    shuffle=False
    shuffle_bufer_size=0
    prefetch=4

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

    # Define the dataset from the list of sharded, shuffled files
    data = tf.data.Dataset.from_tensor_slices(filenames)
    data = data.shard(num_shards=n_shards, index=shard)
    if shuffle:
        data = data.shuffle(len(filenames), reshuffle_each_iteration=True)

    # Parse TFRecords
    parse_data = partial(_parse_data, shape=sample_shape, apply_log=apply_log)
    data = data.apply(tf.data.TFRecordDataset).map(parse_data, num_parallel_calls=4)

    # Parallelize reading with interleave - no benefit?
    #data = data.interleave(
    #    lambda x: tf.data.TFRecordDataset(x).map(parse_data, num_parallel_calls=1),
    #    cycle_length=4
    #)
    # Localized sample shuffling (note: imperfect global shuffling).
    # U
    if shuffle and shuffle_buffer_size > 0:
        data = data.shuffle(shuffle_buffer_size)

    # Construct batches
    data = data.repeat(n_epochs)
    data = data.batch(batch_size, drop_remainder=True)

    # Prefetch to device
    return data.prefetch(prefetch)

  def train_input_fn():
    return _data_input_fn('train')

  def eval_input_fn():
    return _data_input_fn('validation')

  # Train and evaluate model.
  val_dir=os.path.join(FLAGS.data_dir, 'validation')
  
  comm.Barrier()
  for _ in range(FLAGS.train_epochs // FLAGS.epochs_between_evals):
    #cosmo_classifier.train(input_fn=train_input_fn, hooks=None)
    cosmo_classifier.train(input_fn=train_input_fn_dmy, hooks=None)
    eval_results = cosmo_classifier.evaluate(input_fn=eval_input_fn)
    print("\nEvaluation results:\n\t%s\n" % eval_results)


def main(_):
  tf.enable_eager_execution()
  run_cosmo()


if __name__ == "__main__":
  #tf.disable_v2_behavior()
  tf.config.threading.set_inter_op_parallelism_threads(1)
  tf.enable_eager_execution()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
