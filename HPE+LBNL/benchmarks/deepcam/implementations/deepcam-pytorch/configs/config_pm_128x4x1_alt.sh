#!/bin/bash

# data directory
source $(dirname ${BASH_SOURCE[0]})/config_pm_common.sh 

# hyperparameters
export LOCAL_BATCH_SIZE=1
export START_LR=0.0062
export OPTIMIZER="MixedPrecisionLAMB"
export LR_SCHEDULE_TYPE="cosine_annealing"
export LR_T_MAX="2600"
export LR_ETA_MIN="0.0"
export LR_WARMUP_STEPS=400
export LR_WARMUP_FACTOR=1.
export WEIGHT_DECAY=0.01
export BATCHNORM_GROUP_SIZE=2

# data parameters
export SHUFFLE_MODE="global"
export DATA_FORMAT="dali-es-gpu"
export DATA_OVERSAMPLING_FACTOR=1
export PRECISION_MODE="amp"
export LOCAL_VALIDATION_BATCH_SIZE=8
export MAX_THREADS=8

# misc args
export ADDITIONAL_ARGS="${ADDITIONAL_ARGS} --disable_comm_overlap --enable_graph --enable_groupbn"
#export ADDITIONAL_ARGS="${ADDITIONAL_ARGS} --enable_jit --disable_comm_overlap --enable_graph --enable_groupbn"

# system parameters
export DGXNNODES=128
WALLTIME_MINUTES=7
export WALLTIME=$(( 15 + (${NEXP} * ${WALLTIME_MINUTES}) ))
#export SBATCH_ARRAY_INX="1-5%1"
