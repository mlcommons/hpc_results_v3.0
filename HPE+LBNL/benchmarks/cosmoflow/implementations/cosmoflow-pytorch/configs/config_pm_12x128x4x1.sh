source $(dirname ${BASH_SOURCE[0]})/config_pm_common.sh

export STAGING_DIR="/tmp"
export NUM_INSTANCES=12

export CONFIG_FILE="submission_pm_128x4x1.yaml"

## System run parms
export DGXNNODES=1536
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

WALLTIME_MINUTES=25
export WALLTIME=$(( 15 + (${NEXP:-1} * ${WALLTIME_MINUTES}) ))
