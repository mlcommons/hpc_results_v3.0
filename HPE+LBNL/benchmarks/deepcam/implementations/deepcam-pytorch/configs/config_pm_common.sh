#!/bin/bash

# data directory
export DATADIR=/pscratch/sd/s/sfarrell/deepcam-hpc-v1.0/data/All-Hist/numpy

# this should never be exceeded by any benchmark
export MAX_EPOCHS=50

# this is for some global parameters:
export ADDITIONAL_ARGS="--disable_tuning"

# auxiliary parameters
export LOGGING_FREQUENCY=0

# run parameters
export NEXP=1

# system parameters
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
