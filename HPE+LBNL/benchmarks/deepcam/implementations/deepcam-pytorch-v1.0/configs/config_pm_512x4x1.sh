#!/bin/bash

# hyperparameters
export LOCAL_BATCH_SIZE=1
export START_LR=0.0055
export OPTIMIZER="LAMB"
export LR_SCHEDULE_TYPE="multistep"
export LR_MILESTONES="800"
export LR_DECAY_RATE="0.1"
export LR_WARMUP_STEPS=400
export LR_WARMUP_FACTOR=1.
export WEIGHT_DECAY=0.01
export BATCHNORM_GROUP_SIZE=2

# data parameters
export SHUFFLE_MODE="global"
export DATA_FORMAT="dali-es"
export DATA_OVERSAMPLING_FACTOR=2
export PRECISION_MODE="amp"
export LOCAL_VALIDATION_BATCH_SIZE=8

# output parameters
#export OUTPUT_ROOT=/results/best

# auxiliary parameters
export LOGGING_FREQUENCY=10

# misc args
#export ADDITIONAL_ARGS="--disable_comm_overlap --enable_graph"
export ADDITIONAL_ARGS="--enable_jit --disable_comm_overlap --enable_graph"

# run parameters
export NEXP=1

# system parameters
export DGXNGPU=4
export DGXNNODES=512
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:30:00
