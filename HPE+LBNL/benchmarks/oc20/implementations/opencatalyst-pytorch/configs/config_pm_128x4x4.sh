#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_pm_common.sh

# hyperparameters
export BATCH_SIZE=4
export LR_INITIAL=0.0016
export WARMUP_STEPS=3908
export LR_MILESTONES="23448 31264"
export NUM_INSTANCES=1

# system parameters
export DGXNNODES=128
export SBATCH_CONSTRAINT="gpu&hbm80g"
WALLTIME_MINUTES=90
export WALLTIME=$(( 10 + (${NEXP} * ${WALLTIME_MINUTES}) ))
export SBATCH_ARRAY_INX="1-5%1"
