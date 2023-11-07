#!/bin/bash

export DGXNGPU=4
# export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export CHECK_COMPLIANCE=0

# training and validation dataset ranges:
export TRAIN_MAX_PDB_RELEASE_DATE="2021-09-16"
export VAL_MIN_CAMEO_SUBMISSION_DATE="2021-09-17"
export VAL_MAX_CAMEO_SUBMISSION_DATE="2021-12-11"

# target metric value:
export TARGET_AVG_LDDT_CA_VALUE=0.8

# numerical precision:
export PRECISION="bf16"

# training length:
export NUM_TRAIN_ITERS=2000

# validation frequency:
export VAL_EVERY_ITERS=40

# local batch size:
export LOCAL_BATCH_SIZE=1

# dynamic_axial_parallelism:
export DAP_SIZE=0

# learning rate schedule:
export BASE_LR=0.001
export WARMUP_LR_INIT=0.00001
export WARMUP_LR_ITERS=0

# dataloader settings:
export INITIAL_TRAINING_DATALOADER_TYPE="InitialTrainingDataloaderPQ"
export NUM_TRAIN_DATALOADER_WORKERS=14
export NUM_VAL_DATALOADER_WORKERS=2

# async evaluation:
export NUM_ASYNC_VAL_RANKS=0

# optional argument to use specific samples only:
export USE_ONLY_PDB_CHAIN_IDS_ARG=""

# optional argument to denote apilogs generation:
export APILOGS_ARG=""

export DATADIR=/pscratch/sd/s/sfarrell/openfold-ref/data/pdb_data
export CHECKPOINT_DIR=${SCRATCH}/openfold-ref/mlperf_hpc_openfold_resumable_checkpoint_v2.pt

# enable sharp
# export SBATCH_NETWORK=sharp

# benchmark type
export MLPERF_BENCHMARK_TYPE="TimeToTrain"
