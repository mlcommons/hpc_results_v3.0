source $(dirname ${BASH_SOURCE[0]})/config_pm_common.sh

export NCCL_DEBUG=INFO

export STAGING_DIR="/tmp"
export NUM_INSTANCES=2

#export DATADIR=/global/cfs/cdirs/nstaff/sfarrell/mlperf/cosmoflow/data/hpc_v2.0_gzip_4k
export DATADIR=/pscratch/sd/s/sfarrell/cosmoflow-benchmark/data/hpc_v2.0_gzip_1k
export CONFIG_FILE="submission_pm_128x4x1.yaml"

## System run parms
export DGXNNODES=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

WALLTIME_MINUTES=0
export WALLTIME=$(( 15 + (${NEXP:-1} * ${WALLTIME_MINUTES}) ))
export SBATCH_QOS=debug
