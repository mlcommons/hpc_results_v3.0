"""
This module contains some utility callbacks for Keras training.
"""

# System
from time import time
import math

# Externals
import tensorflow as tf
from mlperf_logging import mllog
import logging

class MLPerfLoggingCallback(tf.keras.callbacks.Callback):
    """A Keras Callback for logging MLPerf results"""
    def __init__(self):
        self.mllogger = mllog.get_mllogger()

    def on_epoch_begin(self, epoch, logs={}):
        self.mllogger.start(key=mllog.constants.EPOCH_START,
                            metadata={'epoch_num': epoch})
        self._epoch = epoch

    def on_train_batch_end(self, batch, logs={}):
        if batch%10==0:
            logging.info('batch_end(train) {}'.format(batch))

    def on_test_batch_end(self, batch, logs={}):
        if batch%10==0:
            logging.info('batch_end(test) {}'.format(batch))

    def on_test_begin(self, logs):
        self.mllogger.start(key=mllog.constants.EVAL_START,
                            metadata={'epoch_num': self._epoch})

    def on_test_end(self, logs):
        self.mllogger.end(key=mllog.constants.EVAL_STOP,
                          metadata={'epoch_num': self._epoch})

    def on_epoch_end(self, epoch, logs={}):
        self.mllogger.end(key=mllog.constants.EPOCH_STOP,
                          metadata={'epoch_num': epoch})
        val_mae = logs.get('val_mae', math.nan)
        self.mllogger.event(key='eval_error', value=val_mae,
                            metadata={'epoch_num': epoch})

class TimingCallback(tf.keras.callbacks.Callback):
    """A Keras Callback which records the time of each epoch"""
    def __init__(self):
        self.times = []
    
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time()

    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time() - self.starttime
        self.times.append(epoch_time)
        logs['time'] = epoch_time

#class LearningRateScheduleCallback(tf.keras.callbacks.Callback):
#    def __init__(self, multiplier,
#                 start_epoch=0, end_epoch=None,
#                 momentum_correction=True):
#        super().__init__()
#        self.start_epoch = start_epoch
#        self.end_epoch = end_epoch
#        self.momentum_correction = momentum_correction
#        self.initial_lr = None
#        self.restore_momentum = None

from tensorflow.python.client import timeline

class ProfilingCallback(tf.keras.callbacks.Callback):
    """A Keras Callback which records the time of each epoch"""
    def __init__(self, run_metadata, prefix, step_num):
        self.run_metadata = run_metadata
        self.out_prefix = prefix
        self.step_num = step_num

    def on_batch_end(self, batch, logs = None):
        if batch == self.step_num:
            trace = timeline.Timeline(step_stats=self.run_metadata.step_stats)
            with open('{}_{}.json'.format(self.out_prefix, batch) , 'w') as f:
                f.write(trace.generate_chrome_trace_format())


# from https://stackoverflow.com/questions/53500047/stop-training-in-keras-when-accuracy-is-already-1-0
class TerminateOnBaseline(tf.keras.callbacks.Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='val_mae', baseline=0.124):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc < self.baseline:
                self.model.stop_training = True

class TerminateOnDivergence(tf.keras.callbacks.Callback):
    """Callback that terminates training when monitored value diverges
    """
    def __init__(self, monitor='val_mae', baseline=1.0):
        super(TerminateOnDivergence, self).__init__()
        self.monitor = monitor
        self.baseline = baseline
        self.error_count = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline or acc != acc:
                self.error_count += 1
            else:
                self.error_count = 0
            if self.error_count >= 3:
                raise ValueError('diverged')
