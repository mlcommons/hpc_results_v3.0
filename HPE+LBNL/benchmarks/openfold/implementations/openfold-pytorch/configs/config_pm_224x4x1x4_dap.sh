#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_pm_common.sh

# dynamic_axial_parallelism:
export DAP_SIZE=4

# dataloader settings:
export NUM_VAL_DATALOADER_WORKERS=4

# async evaluation:
export NUM_ASYNC_VAL_RANKS=24

# system parameters:
export DGXNNODES=224
WALLTIME_MINUTES=40
export WALLTIME=$(( 10 + (${NEXP} * ${WALLTIME_MINUTES}) ))
