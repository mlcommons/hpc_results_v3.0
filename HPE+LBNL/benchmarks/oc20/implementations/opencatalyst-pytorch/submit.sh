#!/bin/bash

CONFIG=${1:-configs/config_pm_2x4x4_test.sh}
echo "Using config ${CONFIG}"
source ${CONFIG}

sbatch -N $DGXNNODES -t $WALLTIME run_pm.sub
