#!/bin/bash

export DGXNGPU=8
export LR_GAMMA=0.1
export WARMUP_FACTOR=0.2
export EVAL_BATCH_SIZE=128
export ITERATIONS=0

export EVAL_NODES=0
export MAX_EPOCHS=45
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )
