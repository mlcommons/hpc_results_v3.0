#!/bin/bash

CONFIG=${1:-configs/config_pm_224x4x1x4_dap.sh}
echo "Using config ${CONFIG}"
source ${CONFIG}

sbatch -N $DGXNNODES -A m4291 -t $WALLTIME run_pm.sub
#sbatch -N 2 -q debug -A dasrepo -t 5:00 run_pm.sub
