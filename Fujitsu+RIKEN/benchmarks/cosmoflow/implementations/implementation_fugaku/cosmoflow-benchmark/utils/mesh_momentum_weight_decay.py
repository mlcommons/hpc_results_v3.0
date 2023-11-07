import os
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from mesh_tensorflow import ops as mtf_ops

class MomentumWD(mtf.optimize.Optimizer):
  """SGD with momentum and weight decay."""

  def __init__(self, learning_rate, momentum, weight_decay=0.0):
    self._lr = learning_rate
    self._momentum = momentum
    self._weight_decay = weight_decay

  @property
  def lr(self):
    return self._lr

  @property
  def momentum(self):
    return self._momentum

  @property
  def weight_decay(self):
    return self._weight_decay

  def apply_grad(self, grad, var):
    if grad is None:
      tf.logging.warning("Gradient is None for variable %s" % var.name)
      return []

    decay_rate = 0.0
    if var.name.startswith("conv"):
      scale = mtf_ops.get_model_size() / mtf_ops.get_world_size()
    elif var.name.startswith("hidden"):
      scale = 1.0 / mtf_ops.get_world_size()
      if "kernel" in var.name and "l2" in var.name:
          decay_rate = self.weight_decay
    else:
      raise Exception("grad scale of {} is not found".format(var.name))

    grad *= scale
    grad += var.value * decay_rate

    updates = []
    v = mtf.get_variable(
        var.mesh, var.name + "_momentum_v", var.shape,
        dtype=var.dtype, initializer=tf.zeros_initializer(), trainable=False)

    get_hist = int(os.environ.get('MTF_HISTOGRAM', 0)) != 0
    if get_hist:
      grad2 = mtf.get_variable(
              var.mesh, var.name + "_grad", var.shape,
              dtype=var.dtype, initializer=tf.zeros_initializer(), trainable=False)  # to visualize grad

    with tf.variable_scope(var.name + "/sgd_momentum"):
      vel = grad * self.lr + v * self.momentum
      updates.append(mtf.assign(v, vel))
      updates.append(mtf.assign_sub(var, vel))
      if get_hist:
        updates.append(mtf.assign(grad2, grad)) # to visualize grad

    return updates
