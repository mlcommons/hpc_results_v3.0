#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_pm_common.sh

# dynamic_axial_parallelism:
export DAP_SIZE=4

export NEXP=1

# system parameters:
export DGXNNODES=128
WALLTIME_MINUTES=70

export WALLTIME=$(( 10 + (${NEXP} * ${WALLTIME_MINUTES}) ))
