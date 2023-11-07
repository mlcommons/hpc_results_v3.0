source $(dirname ${BASH_SOURCE[0]})/config_pm_common.sh

export CONFIG_FILE="submission_dgxa100_64x8x1.yaml"

## System run parms
export DGXNNODES=128
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

WALLTIME_MINUTES=15
export WALLTIME=$(( 15 + (${NEXP:-1} * ${WALLTIME_MINUTES}) ))
#export SBATCH_ARRAY_INX="1-10%1"
