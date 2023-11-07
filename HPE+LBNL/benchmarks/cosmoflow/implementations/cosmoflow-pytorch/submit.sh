#!/bin/bash

CONFIG=${1:-configs/config_pm_2x4x1_test.sh}
echo "Using config ${CONFIG}"
source ${CONFIG}

sbatch -N $DGXNNODES -t $WALLTIME ${EXTRA_SBATCH_ARGS} run_pm.sub
