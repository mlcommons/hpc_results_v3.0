#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

ENABLE_IB_BINDING=${ENABLE_IB_BINDING:-1}

# Updated SHS libs from HPE for testing
export LD_LIBRARY_PATH=/pscratch/sd/p/pharring/shspatch/libshs22_Aug7debug:$LD_LIBRARY_PATH
export FI_CXI_COMPAT=2
# Force NCCL to use simple protocol rather than low-latency
export NCCL_PROTO=simple

# Sporadic perf drop fixes
#export FI_CXI_RX_MATCH_MODE=software
#export FI_CXI_RDZV_PROTO=alt_read

# NCCL OFI plugin
#source /pscratch/sd/s/sfarrell/nccl-ofi/nccl-ofi-plugin-in-cont/env_nccl.sh

PARAMS=(
    --train_dataset ${TRAIN_DATASET:-oc20_data/s2ef/2M/train}
    --val_dataset ${VAL_DATASET:-oc20_data/s2ef/all/val_id}
    --data_target ${DATA_TARGET:-/dev/shm}
    --batch_size ${BATCH_SIZE}
    --eval_batch_size ${EVAL_BATCH_SIZE}
    --lr_initial ${LR_INITIAL}
    --warmup_steps ${WARMUP_STEPS}
    --warmup_factor ${WARMUP_FACTOR}
    --lr_milestones ${LR_MILESTONES}
    --lr_gamma ${LR_GAMMA}
    --instances ${NUM_INSTANCES}
    --nodes_for_eval ${EVAL_NODES}
    --max_epochs ${MAX_EPOCHS}
    --iterations ${ITERATIONS}
    --seed ${SEED}
)

IB_BIND=''
if [[ "${SLURM_JOB_NUM_NODES}" -gt 1 && "${ENABLE_IB_BINDING}" -eq 1 ]]; then
  IB_BIND='--ib=single'
fi
#BIND="bindpcie --cpu=exclusive,nosmt ${IB_BIND} --"

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
${LOGGER:-} ${BIND} python main.py "${PARAMS[@]}"; ret_code=$?
set +x

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="OC20_HPC"
echo "RESULT,$result_name,,$result,nvidia,$start_fmt"
