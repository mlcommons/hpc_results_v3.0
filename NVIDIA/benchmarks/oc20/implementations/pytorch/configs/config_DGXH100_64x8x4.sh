#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_DGXH100_common.sh

# hyperparameters
export BATCH_SIZE=4
export LR_INITIAL=0.0016
export WARMUP_STEPS=3908
export LR_MILESTONES="23448 31264"
export NUM_INSTANCES=1
export SBATCH_NETWORK=sharp

# system parameters
export DGXNNODES=64
WALLTIME_MINUTES=20
export WALLTIME=$(( 10 + (${NEXP} * ${WALLTIME_MINUTES}) ))
