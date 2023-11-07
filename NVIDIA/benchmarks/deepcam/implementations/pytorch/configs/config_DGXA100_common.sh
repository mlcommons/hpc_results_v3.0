#!/bin/bash

# this should never be exceeded by any benchmark
export MAX_EPOCHS=50

# this is for some global parameters:
export ADDITIONAL_ARGS="--disable_tuning"

# auxiliary parameters
export LOGGING_FREQUENCY=0

# direct io settings
export DALI_ODIRECT_ALIGNMENT=4096
export DALI_ODIRECT_LEN_ALIGNMENT=4096

# run parameters
export NEXP="${NEXP:-10}"

# system parameters
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export BASE_COMP_CLOCK=1410
export BASE_MEM_CLOCK=1593
