"""
Hardware/device configuration
"""

# System
import os

# Externals
import tensorflow as tf

def configure_session(intra_threads=48, inter_threads=1,
                      blocktime=1, affinity='granularity=fine,compact,1,0',
                      gpu=None, seed=-1):
    """Sets the thread knobs in the TF backend"""
    os.environ['KMP_BLOCKTIME'] = str(blocktime)
    os.environ['KMP_AFFINITY'] = affinity
    os.environ['OMP_NUM_THREADS'] = str(intra_threads)
    config = tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=inter_threads,
        intra_op_parallelism_threads=intra_threads,
        allow_soft_placement=True,
    )
    if gpu is not None:
        config.gpu_options.visible_device_list = str(gpu)
        config.gpu_options.per_process_gpu_memory_fraction=0.9 # reduce GPU memory usage to keep memory for NCCL

    if seed != -1:
        tf.random.set_seed(seed)
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
