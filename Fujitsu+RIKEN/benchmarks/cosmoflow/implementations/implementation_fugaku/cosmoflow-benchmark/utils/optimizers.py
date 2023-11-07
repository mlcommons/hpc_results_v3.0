"""
Utilty code for constructing optimizers and scheduling learning rates.
"""

# System
import math
from functools import partial

# Externals
from tensorflow import keras
#import horovod.tensorflow.keras as hvd
from mlperf_logging import mllog
import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf

from .momentum_lars import MomentumLARS
from .mesh_momentum_weight_decay import MomentumWD

def _lr_schedule(epoch, init_lr, peak_lr, n_warmup_epochs, decay_schedule={}):
    """Learning rate schedule function.

    Gives the learning rate as a function of epoch according to
    additional settings:
        base_lr: baseline unscaled learning rate at beginning of training.
        peak_lr: scaled learning at end of warmup period
        n_warmup_epochs: number of linear warmup epochs
        decay_schedule: a dict of epoch number -> decay factor
    """
    # Linear LR warmup
    if epoch < n_warmup_epochs:
        return epoch * (peak_lr - init_lr) / n_warmup_epochs + init_lr
    else:
        decay_name = decay_schedule['name']

        if decay_name == 'step':
            decay_steps = decay_schedule.copy()
            decay_steps.pop('name')
            # Find the most recent decay factor
            decay_factor = 1.
            decay_epoch = 0
            for e, d in decay_steps.items():
                if e >= decay_epoch and e < epoch:
                    decay_epoch, decay_factor = e, d
            return peak_lr * decay_factor
        elif decay_name == 'poly':
            n_decay_epochs = decay_schedule['n_decay_epochs']
            end_lr_factor = decay_schedule['end_factor']
            power = decay_schedule['power']
            decay_epoch = min(epoch - n_warmup_epochs, n_decay_epochs)
            end_lr = peak_lr * end_lr_factor
            return ((peak_lr - end_lr) * (1 - decay_epoch / n_decay_epochs)**power) + end_lr
        elif decay_name == 'cos':
            n_decay_epochs = decay_schedule['n_decay_epochs']
            end_lr_factor = decay_schedule['end_factor']
            decay_epoch = min(epoch - n_warmup_epochs, n_decay_epochs)
            end_lr = peak_lr * end_lr_factor
            return ((peak_lr - end_lr) * (0.5 * (1 + math.cos(math.pi * decay_epoch / n_decay_epochs)))) + end_lr
        elif decay_name == 'htd':
            n_decay_epochs = decay_schedule['n_decay_epochs']
            end_lr_factor = decay_schedule['end_factor']
            U = decay_schedule['U']
            L = decay_schedule['L']
            decay_epoch = min(epoch - n_warmup_epochs, n_decay_epochs)
            end_lr = peak_lr * end_lr_factor
            return ((peak_lr - end_lr) * (0.5 * (1 - math.tanh(L+(U-L)*(decay_epoch / n_decay_epochs))))) + end_lr
        elif decay_name == 'natan':
            n_decay_epochs = decay_schedule['n_decay_epochs']
            end_lr_factor = decay_schedule['end_factor']
            turn_epoch = decay_schedule['turn_epoch']
            decay_epoch = min(epoch - n_warmup_epochs, n_decay_epochs)
            end_lr = peak_lr * end_lr_factor
            df = decay_epoch - turn_epoch
            end_df = n_decay_epochs - turn_epoch
            def f(x):
                if x > 0:
                    x *= 2
                return 0.001 * (x * x * x) + 0.2 * x
            def natan(x):
                return (0.5 - math.atan(f(x)) / math.pi)
            r = natan(df)
            r_min = natan(end_df)
            r = (r - r_min) / (1 - r_min)
            return (peak_lr - end_lr) * r + end_lr
        else:
            raise Exception('decay name is not specified or not supported')

def poly_decay(step, decay_steps, end_lr_factor, power):
    step_ = tf.cast(tf.minimum(step, decay_steps), tf.float32)
    return (1.0 - end_lr_factor) * (1.0 - tf.math.pow(step_ / decay_steps, power)) + end_lr_factor

def step_decay(step, decays):
    scale = 1.0
    for s, d in decays:
        scale = tf.cond(step >= s,
                        lambda: d,
                        lambda: scale)
    return scale

def cos_decay(step, decay_steps, end_lr_factor):
    step_ = tf.cast(tf.minimum(step, decay_steps), tf.float32)
    return ((1 - end_lr_factor) * (0.5 * (1.0 + tf.math.cos(math.pi * step_ / decay_steps)))) + end_lr_factor

def htd_decay(step, decay_steps, end_lr_factor, U, L):
    step_ = tf.cast(tf.minimum(step, decay_steps), tf.float32)
    return ((1.0 - end_lr_factor) * (0.5 * (1.0 - tf.math.tanh(L+(U-L)*(step_ / decay_steps))))) + end_lr_factor

def _lr_schedule_mod(step, init_lr, peak_lr, warmup_steps, decay_fn, change_freq=1):
    """Learning rate schedule function.

    Gives the learning rate as a function of epoch according to
    additional settings:
        base_lr: baseline unscaled learning rate at beginning of training.
        peak_lr: scaled learning at end of warmup period
        n_warmup_epochs: number of linear warmup epochs
        decay_schedule: a dict of epoch number -> decay factor
    """
    # Linear LR warmup

    step = tf.math.floor(step / change_freq)
    #step = tf.Print(step, [step], message="step: ")

    lr = tf.cond(step < warmup_steps,
                 lambda: (peak_lr - init_lr) * (tf.cast(step, tf.float32) / warmup_steps) + init_lr,
                 lambda: peak_lr * decay_fn(step - warmup_steps))

    #lr = tf.Print(lr, [lr], message="LR: ")

    return lr

def get_lr_schedule(base_lr, global_batch_size, epoch_steps, base_batch_size=None,
                    scaling=None, n_warmup_epochs=0, change_freq=None, decay_schedule={}, is_root=True):
    """Get the learning rate schedule function"""
    if scaling == 'linear':
        scale_factor = global_batch_size / base_batch_size
    elif scaling == 'sqrt':
        scale_factor = math.sqrt(global_batch_size / base_batch_size)
    else:
        scale_factor = 1.;
    peak_lr = base_lr * scale_factor
    init_lr = base_lr

    # MLPerf logging
    # NOTE: there is currently a confusing mismatch between the parameter
    # naming convention in this implementation and MLPerf's hyperparameter
    # conventions. Here we define base LR to be the LR at a baseline batch
    # size and the "peak" LR to be the value scaled according to current batch
    # size. We will leave things as-is for now.
    if is_root:
        mllogger = mllog.get_mllogger()
        mllogger.event(key=mllog.constants.OPT_BASE_LR, value=peak_lr)
        mllogger.event(key=mllog.constants.OPT_LR_WARMUP_EPOCHS, value=n_warmup_epochs)
        mllogger.event(key=mllog.constants.OPT_LR_WARMUP_FACTOR, value=scale_factor)

        decay_name = decay_schedule['name']
        if decay_name == 'step':
            decay_steps = decay_schedule.copy()
            decay_steps.pop('name')
            mllogger.event(key=mllog.constants.OPT_LR_DECAY_BOUNDARY_EPOCHS,
                           value=sorted(decay_steps.keys()))
            mllogger.event(key=mllog.constants.OPT_LR_DECAY_FACTOR,
                           value=max(decay_steps.values()) if len(decay_steps)>0 else 1)

    if change_freq is None:
        change_freq = epoch_steps  # 1 for step level decay, epoch_steps for epoch level decay

    step_factor = epoch_steps / change_freq
    decay_name = decay_schedule['name']
    if decay_name == 'step':
        decays = []
        for e, d in decay_schedule.items():
            if (type(e) is int):
                decays.append(((e-n_warmup_epochs+1)*step_factor, d)) # +1 is for the same behavior as the reference implementation
        decays.sort(key=lambda x: x[0])
        decay_fn = partial(step_decay, decays=decays)
    elif decay_name == 'poly':
        n_decay_epochs = decay_schedule['n_decay_epochs']
        end_lr_factor = decay_schedule['end_factor']
        power = decay_schedule['power']
        decay_fn = partial(poly_decay, decay_steps=n_decay_epochs * step_factor, end_lr_factor=end_lr_factor, power=power)
    elif decay_name == 'cos':
        n_decay_epochs = decay_schedule['n_decay_epochs']
        end_lr_factor = decay_schedule['end_factor']
        decay_fn = partial(cos_decay, decay_steps=n_decay_epochs * step_factor, end_lr_factor=end_lr_factor)
    elif decay_name == 'htd':
        n_decay_epochs = decay_schedule['n_decay_epochs']
        end_lr_factor = decay_schedule['end_factor']
        U = decay_schedule['U']
        L = decay_schedule['L']
        decay_fn = partial(htd_decay, decay_steps=n_decay_epochs * step_factor, end_lr_factor=end_lr_factor, U=U, L=L)
    elif decay_name in ('natan'):
        raise Exception('{} is not implemented'.format(decay_name))
    else:
        decay_fn = lambda step: 1.0

    lr_schedule = partial(_lr_schedule_mod, init_lr=init_lr, peak_lr=peak_lr,
                          warmup_steps=n_warmup_epochs * step_factor,
                          decay_fn=decay_fn, change_freq=change_freq)
    return lr_schedule

def get_optimizer(name, distributed=False, is_root=True, **opt_args):
    """Configure the optimizer"""

    # MLPerf logging
    if is_root:
        mllogger = mllog.get_mllogger()
        mllogger.event(key=mllog.constants.OPT_NAME, value=name)
        mllogger.event(key=mllog.constants.OPT_WEIGHT_DECAY, value=opt_args['weight_decay'] / 2)

    # Construct the optimizer
    if name == 'SGD':
        OptType = MomentumWD
    elif name in ('LAMB', 'RAdam', 'LARS'):
        raise Exception('Optimizer %s is not supported' % name)
    else:
        OptType = getattr(keras.optimizers, name)
    opt = partial(OptType, **opt_args)

    # Distributed optimizer wrapper
    #if distributed:
        #opt = hvd.DistributedOptimizer(opt)
        #opt = hvd.DistributedOptimizer(opt, compression=hvd.Compression.fp16)

    return opt
