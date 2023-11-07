source $(dirname ${BASH_SOURCE[0]})/config_pm_common.sh

## DL params
export BATCHSIZE="1"
export APPLY_LOG_TRANSFORM="1"
export INIT_LR="0.001"
export LR="0.012"
export NUMEPOCHS="90"

export WARMUP_EPOCHS="0"
export LRSCHED_EPOCHS="32 64"

export APPLY_SHUFFLE="1"
export APPLY_PRESHUFFLE="1"
export GRAD_PREDIV_FACTOR="1.0"
export DATA_SHARD_MULTIPLIER="2"

export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=999
export NCCL_COLLNET_ENABLE=0 # disable SHARP

## System run parms
export DGXNNODES=256
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:30:00
