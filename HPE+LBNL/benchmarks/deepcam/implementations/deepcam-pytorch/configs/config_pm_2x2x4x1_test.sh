#!/bin/bash

# common parameters
source $(dirname ${BASH_SOURCE[0]})/config_pm_common.sh

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
export MAX_EPOCHS=8
export TRAINING_INSTANCE_SIZE=8

# data parameters
export DATADIR=/pscratch/sd/s/sfarrell/deepcam-hpc-v1.0/data/256-dataset/
export SHUFFLE_MODE="global"
export DATA_FORMAT="dali-numpy"
export PRECISION_MODE="amp"
export LOCAL_VALIDATION_BATCH_SIZE=8

# staging parameters
export STAGE_DIR_PREFIX="/tmp"
export STAGE_BATCH_SIZE=8
export STAGE_MODE="global"
export STAGE_VERIFY=0
export STAGE_FULL_DATA_PER_NODE=0
export STAGE_USE_DIRECT_IO=0
export STAGE_NUM_READ_WORKERS=2
export STAGE_NUM_WRITE_WORKERS=8

# misc args
export ADDITIONAL_ARGS="${ADDITIONAL_ARGS} --disable_comm_overlap --enable_graph --enable_groupbn"
#export ADDITIONAL_ARGS="${ADDITIONAL_ARGS} --enable_jit --disable_comm_overlap --enable_graph --enable_groupbn"

# system parameters
export NUM_INSTANCES=2
export NCCL_DEBUG=INFO
export DGXNNODES=4
export WALLTIME=00:15:00
