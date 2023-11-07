"""Configurable model specification for Mesh CosmoFlow"""

import numpy as np
import mesh_tensorflow as mtf
from functools import partial, reduce
from operator import mul
import logging
import tensorflow.compat.v1 as tf

dep_blocks_per_proc = 1
row_blocks_per_proc = 1

def build_model(input_shape, target_size,
                conv_size=16, kernel_size=3, n_conv_layers=5,
                fc1_size=128, fc2_size=64,
                hidden_activation='LeakyReLU',
                pooling_type='MaxPool3D',
                dropout=0,
                use_conv_bias=True,
                optimizer=None,
                lr_scheduler=None,
                loss=None,
                metrics=None,
                batch_size=1,
                mesh_shape=(1,1),
                num_mpiar_tensors=999,
                seed=None,
                save_conf={'enabled':False}):

    model = partial(model_fn, dropout=dropout, optimizer=optimizer, lr_scheduler=lr_scheduler,
                    batch_size=batch_size, mesh_shape=mesh_shape, seed=seed, save_conf=save_conf,
                    kernel_size=kernel_size, num_mpiar_tensors=num_mpiar_tensors)

    return model


def cosmo_model(image, labels, mesh, mode,
                batch_size, mesh_shape, dropout, seed, kernel_size):
  """The model.

  Args:
    image: tf.Tensor with shape [batch, 28*28]
    labels: a tf.Tensor with shape [batch] and dtype tf.int32
    mesh: a mtf.Mesh

  Returns:
    logits: a mtf.Tensor with shape [batch, 10]
    loss: a mtf.Tensor with shape []
  """
  mesh_height, mesh_width = mesh_shape
  batch_dim = mtf.Dimension("batch", batch_size)
  dep_blocks_dim = mtf.Dimension("dep_blocks", dep_blocks_per_proc)
  row_blocks_dim = mtf.Dimension("row_blocks", row_blocks_per_proc)
  col_blocks_dim = mtf.Dimension("col_blocks", 1)
  deps_dim = mtf.Dimension("deps_size", 128 // mesh_height)
  rows_dim = mtf.Dimension("rows_size", 128 // mesh_width)
  cols_dim = mtf.Dimension("cols_size", 128)

  channel_dim = mtf.Dimension("one_channel", 4)

  x = mtf.import_tf_tensor(
      #mesh, tf.reshape(image, [FLAGS.batch_size, 4, 32, 4, 32, 1, 128, 4]),
      mesh, tf.reshape(image, [batch_size, dep_blocks_per_proc, 128//mesh_height,
                               row_blocks_per_proc, 128//mesh_width, 1, 128, 4]),
      mtf.Shape(
          [batch_dim, dep_blocks_dim, deps_dim, row_blocks_dim, rows_dim,
           col_blocks_dim, cols_dim, channel_dim]))
  x = mtf.transpose(x, [
      batch_dim, dep_blocks_dim, row_blocks_dim, col_blocks_dim,
      deps_dim, rows_dim, cols_dim, channel_dim])

  # add some convolutional layers to demonstrate that convolution works.
  _num_filters=5
  _filter=[kernel_size, kernel_size, kernel_size]
  filters_dim=[]*_num_filters
  for i in range(_num_filters):
    filters_dim.append(mtf.Dimension("filters"+str(i + 1), 32 * (2 ** i)))
    x = mtf.layers.conv3d_with_MPI(
        x, filters_dim[i], filter_size=_filter, strides=[1, 1, 1], padding="SAME",
        d_blocks_dim=dep_blocks_dim, h_blocks_dim=row_blocks_dim,
        filter_initializer=tf.glorot_uniform_initializer(seed=seed),
        name="conv"+str(i))
    if seed is not None: seed += 1
    bias = mtf.get_variable(
           x.mesh,
           "conv_bias"+str(i),
           mtf.Shape([filters_dim[i]]),
           initializer=tf.zeros_initializer(),
           dtype=tf.float32)
    x += bias
    x = mtf.layers.max_pool3d(mtf.leaky_relu(x, alpha=0.3))

  # add some fully-connected dense layers.
  hidden_dim1 = mtf.Dimension("hidden1_l2", 128)
  hidden_dim2 = mtf.Dimension("hidden2_l2", 64)
  hidden_dim3 = mtf.Dimension("hidden3", 4)
  if mesh_height * mesh_width > 1:
      x = mtf.layers.conv3d_to_dense(x, mesh_size=[mesh_height,mesh_width])
  x = mtf.reshape(x, mtf.Shape([x.shape.dims[0],
                                mtf.Dimension("flatten", reduce(mul, [dim.size for dim in x.shape.dims[1:]]))]))
  h1 = mtf.layers.dense(x, hidden_dim1, reduced_dims=x.shape.dims[-1:],
                        kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                        name="hidden1_l2")
  if seed is not None: seed += 1
  h1 = mtf.leaky_relu(h1, alpha=0.3)
  if mode == tf.estimator.ModeKeys.TRAIN:
     h1 = mtf.dropout(h1, rate=dropout)
  h2 = mtf.layers.dense(h1, hidden_dim2, reduced_dims=h1.shape.dims[-1:],
                        kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                        name="hidden2_l2")
  if seed is not None: seed += 1
  h2 = mtf.leaky_relu(h2, alpha=0.3)
  if mode == tf.estimator.ModeKeys.TRAIN:
     h2 = mtf.dropout(h2, rate=dropout)
  h3 = mtf.layers.dense(h2, hidden_dim3, reduced_dims=h2.shape.dims[-1:], activation=mtf.tanh,
                        kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                        name="hidden3")
  if seed is not None: seed += 1
  logits = mtf.multiply(h3, 1.2)
  loss = mtf.square(logits - tf.reshape(labels, (batch_size, 4)))
  loss = mtf.reduce_mean(loss)
  return logits, loss

def model_fn(features, labels, mode, params,
             dropout, optimizer, lr_scheduler, batch_size, mesh_shape, seed, save_conf, kernel_size, num_mpiar_tensors):
  """The model_fn argument for creating an Estimator."""
  logging.info("features = %s labels = %s mode = %s params=%s" %
               (features, labels, mode, params))
  global_step = tf.train.get_global_step()
  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "my_mesh")
  #print("##",features)
  #print("##",labels)
  #print("##",mesh)
  logits, loss = cosmo_model(features, labels, mesh, mode,
                             batch_size=batch_size, mesh_shape=mesh_shape, dropout=dropout, seed=seed, kernel_size=kernel_size)
  mesh_shape_str = 'b1:{};b2:{}'.format(dep_blocks_per_proc, row_blocks_per_proc)
  mesh_shape = mtf.convert_to_shape(mesh_shape_str)
  layout_str = 'dep_blocks:b1;row_blocks:b2'
  layout_rules = mtf.convert_to_layout_rules(layout_str)
  mesh_size = mesh_shape.size
  mesh_devices = [""] * (dep_blocks_per_proc * row_blocks_per_proc)
  mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
      mesh_shape, layout_rules, mesh_devices)

  if mode == tf.estimator.ModeKeys.TRAIN:
    var_grads = mtf.gradients(
        [loss], [v.outputs[0] for v in graph.trainable_variables])
    with mtf.utils.outside_all_rewrites():
        lr = lr_scheduler(global_step)

    optimizer = optimizer(learning_rate=lr)
    update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables, num_mpiar_tensors)

  lowering = mtf.Lowering(graph, {mesh: mesh_impl})

  tf_logits = lowering.export_to_tf_tensor(logits)
  tf.identity(tf_logits, name="train_logits")

  if mode != tf.estimator.ModeKeys.PREDICT:
    tf_loss = lowering.export_to_tf_tensor(loss)
    tf.identity(tf_loss, name="train_loss")
    tf.summary.scalar("loss", tf_loss)
    tf.identity(labels, name="labels")
    tf_mae = tf.metrics.mean_absolute_error(labels=labels, predictions=tf_logits)
    tf.summary.scalar("mae", tf_mae[1])
    tf.identity(tf_mae[1], name="train_mae")
    tf.identity(tf_mae[0], name="train_mae0")

    tf_my_mae = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(tf_logits, labels)))
    tf.identity(tf_my_mae, name="my_train_mae")

  if mode == tf.estimator.ModeKeys.TRAIN:
    tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
    with tf.control_dependencies(tf_update_ops):
      tf_update_ops.append(tf.assign_add(global_step, 1))
    train_op = tf.group(tf_update_ops)

  with mtf.utils.outside_all_rewrites():
      restore_hook = mtf.MtfRestoreHook(lowering)
      if mode == tf.estimator.ModeKeys.TRAIN:
          training_chief_hooks = [restore_hook]
          if save_conf['enabled']:
              saver = tf.train.Saver(
                  tf.global_variables(),
                  sharded=True,
                  max_to_keep=None,
                  keep_checkpoint_every_n_hours=2,
                  defer_build=False, save_relative_paths=True)
              tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
              saver_listener = mtf.MtfCheckpointSaverListener(lowering)
              saver_hook = tf.train.CheckpointSaverHook(
                  save_conf['output_dir'],
                  save_steps=save_conf['save_steps'],
                  saver=saver,
                  listeners=[saver_listener])
              training_chief_hooks.append(saver_hook)

        #mae = tf.metrics.mean_absolute_error(
        #    labels=labels, predictions=tf_logits)

        # Name tensors to be logged with LoggingTensorHook.
        #tf.identity(tf_loss, "cross_entropy")

        # Save accuracy scalar to Tensorboard output.
        #tf.summary.scalar("train_mae", mae[1])

        # restore_hook must come before saver_hook
          return tf.estimator.EstimatorSpec(
              mode,
              loss=tf_loss,
              train_op=train_op,
              training_chief_hooks=training_chief_hooks)

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

          #logging_hook = tf.train.LoggingTensorHook({"val_loss" : tf_loss,
          #                                          "val_mae" : tf_mae[1]},
          #                                          at_end=True)

          return tf.estimator.EstimatorSpec(
              mode,
              loss=tf_loss,
              evaluation_hooks=[restore_hook], #logging_hook
              eval_metric_ops={
                  "mae": tf_mae,
              })
