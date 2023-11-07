#!/bin/bash

export DATADIR="/pscratch/sd/s/sfarrell/optimized-hpc/opencatalyst/data"
export DATA_TARGET="/data" # disables data staging

export DGXNGPU=4
export LR_GAMMA=0.1
export WARMUP_FACTOR=0.2
export EVAL_BATCH_SIZE=128
export ITERATIONS=0

export NEXP=1
export EVAL_NODES=0
export MAX_EPOCHS=45
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
