#!/bin/bash

# Copyright 2023 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Prevent Triton cache collisions across processes on each node.
export TRITON_CACHE_DIR="/dev/shm/triton/job${SLURM_JOBID}/rank${SLURM_PROCID}"
#export DATADIR=/pscratch/sd/s/sfarrell/openfold-ref/data/pdb_data
export CHECKPOINT_DIR=/global/cfs/cdirs/m4291/openfold/checkpoints/mlperf_hpc_openfold_resumable_checkpoint_v2.pt
export SCRIPT_DIR=/pscratch/sd/s/schheda/final/optimized-hpc/openfold/pytorch

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export GROUP_RANK=$SLURM_NODEID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
export MASTER_PORT=29500

ENABLE_IB_BINDING=${ENABLE_IB_BINDING:-1}

PARAMS=(
    --training_dirpath /results
    --pdb_mmcif_chains_filepath ${DATADIR}/pdb_mmcif/processed/chains.csv
    --pdb_mmcif_dicts_dirpath ${DATADIR}/pdb_mmcif/processed/dicts
    --pdb_obsolete_filepath ${DATADIR}/pdb_mmcif/processed/obsolete.dat
    --pdb_alignments_dirpath ${DATADIR}/open_protein_set/processed/pdb_alignments/
    --initialize_parameters_from ${SCRATCH}/openfold-ref/mlperf_hpc_openfold_resumable_checkpoint_v2.pt
    --train_max_pdb_release_date ${TRAIN_MAX_PDB_RELEASE_DATE}
    --val_min_cameo_submission_date ${VAL_MIN_CAMEO_SUBMISSION_DATE}
    --val_max_cameo_submission_date ${VAL_MAX_CAMEO_SUBMISSION_DATE}
    --target_avg_lddt_ca_value ${TARGET_AVG_LDDT_CA_VALUE}
    --precision ${PRECISION}
    --seed ${SEED}
    --num_train_iters ${NUM_TRAIN_ITERS}
    --val_every_iters ${VAL_EVERY_ITERS}
    --local_batch_size ${LOCAL_BATCH_SIZE}
    --dap_size ${DAP_SIZE}
    --base_lr ${BASE_LR}
    --warmup_lr_init ${WARMUP_LR_INIT}
    --warmup_lr_iters ${WARMUP_LR_ITERS}
    --initial_training_dataloader_type ${INITIAL_TRAINING_DATALOADER_TYPE}
    --num_train_dataloader_workers ${NUM_TRAIN_DATALOADER_WORKERS}
    --num_val_dataloader_workers ${NUM_VAL_DATALOADER_WORKERS}
    ${USE_ONLY_PDB_CHAIN_IDS_ARG}
    ${TRAIN_DATALOADER_THREADING}
    --num_async_val_ranks ${NUM_ASYNC_VAL_RANKS}
    ${SAVE_PROCESS_LOGS_FLAG}
    --mlperf_benchmark_type ${MLPERF_BENCHMARK_TYPE}
    ${USE_ONLY_PDB_CHAIN_IDS_ARG}
    ${APILOGS_ARG}
    --distributed
)

IB_BIND=''
#if [[ "${SLURM_JOB_NUM_NODES}" -gt 1 && "${ENABLE_IB_BINDING}" -eq 1 ]]; then
# IB_BIND='--ib=single'
#fi
# BIND="bindpcie --cpu=exclusive ${IB_BIND} --"


# bind command
if [[ "${NERSC_HOST}" == "perlmutter" ]]; then
    BIND="${SCRIPT_DIR}/bind.sh --cpu=${SCRIPT_DIR}/pm_bind_map.sh --mem=${SCRIPT_DIR}/pm_bind_map.sh"
else
    BIND="${SCRIPT_DIR}/bind.sh --cpu=exclusive"
fi

if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  readonly node_rank="${SLURM_NODEID:-0}"
  readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ];
  then
    LOGGER=$LOGGER
  else
    LOGGER=""
  fi
fi
set -x
${LOGGER:-} ${BIND} python train.py "${PARAMS[@]}"; ret_code=$?
set +x
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="OPENFOLD_HPC"
echo "RESULT,$result_name,,$result,nvidia,$start_fmt"
