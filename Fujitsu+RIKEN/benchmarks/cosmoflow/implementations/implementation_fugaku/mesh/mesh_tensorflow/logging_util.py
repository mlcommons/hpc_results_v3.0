# System
import os

from mpi4py import MPI
import tensorflow.compat.v1 as tf

comm_rank = MPI.COMM_WORLD.Get_rank()
def info(msg, *args, **kwargs):
    if comm_rank == 0:
        tf.logging.info(msg, *args, **kwargs)

