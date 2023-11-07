#!/bin/bash

# data directory
source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh 

# hyperparameters
export LOCAL_BATCH_SIZE=8
export START_LR=0.00155
export OPTIMIZER="MixedPrecisionLAMB"
export LR_SCHEDULE_TYPE="cosine_annealing"
export LR_T_MAX="9000"
export LR_ETA_MIN="0.0"
export LR_WARMUP_STEPS=0
export LR_WARMUP_FACTOR=1.
export WEIGHT_DECAY=0.01
export BATCHNORM_GROUP_SIZE=1
export TRAINING_INSTANCE_SIZE=16

# data parameters
export SHUFFLE_MODE="global"
export DATA_FORMAT="dali-numpy"
export PRECISION_MODE="amp"
export LOCAL_VALIDATION_BATCH_SIZE=8

# staging parameter
export STAGE_DIR_PREFIX="/scratch"
export STAGE_BATCH_SIZE=8
export STAGE_MODE="global"
export STAGE_VERIFY=0
export STAGE_FULL_DATA_PER_NODE=0
export STAGE_USE_DIRECT_IO=1
export STAGE_NUM_READ_WORKERS=2
export STAGE_NUM_WRITE_WORKERS=8

# misc args
export ADDITIONAL_SRUN_ARGS="--no-kill"
export ADDITIONAL_ARGS="${ADDITIONAL_ARGS} --enable_graph --disable_comm_overlap"

# number of experiments
export NEXP=1
export NUM_INSTANCES=256

# system parameters
export DGXNNODES=512
export WALLTIME=01:00:00
