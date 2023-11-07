#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_pm_common.sh

# hyperparameters
export BATCH_SIZE=4
export LR_INITIAL=0.0016
export WARMUP_STEPS=3908
export LR_MILESTONES="23448 31264"
export NUM_INSTANCES=1
export MAX_EPOCHS=1

# system parameters
#export SBATCH_NETWORK=""
export NCCL_DEBUG=INFO
export DGXNNODES=2
WALLTIME_MINUTES=10
export WALLTIME=$(( 10 + (${NEXP:-1} * ${WALLTIME_MINUTES}) ))
export CHECK_COMPLIANCE=0
