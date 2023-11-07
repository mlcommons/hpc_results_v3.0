#!/bin/bash
# Copyright (c) 2021-2023 NVIDIA CORPORATION. All rights reserved.
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
#
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
readonly global_rank=${SLURM_PROCID:-}
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"

DGXNGPU=${DGXNGPU:-8}

SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$DGXNGPU}
CONFIG_FILE=${CONFIG_FILE}
NUM_INSTANCES=${NUM_INSTANCES:-1}


PROFILE=${PROFILE:-0}
PROFILE_EXCEL=${PROFILE_EXCEL:-0}
CUDA_PROFILER_RANGE=${CUDA_PROFILER_RANGE:-""}

SEED=${SEED:-0}
ENABLE_IB_BINDING=${ENABLE_IB_BINDING:-1}

PROFILE_ALL_LOCAL_RANKS=${PROFILE_ALL_LOCAL_RANKS:-0}
THR="0.124"

if [[ ${PROFILE} == 1 ]]; then
    THR="0"
fi

DATAROOT="/data"

echo "running benchmark"
export NGPUS=$SLURM_NTASKS_PER_NODE
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}
export DALI_MALLOC_POOL_THRESHOLD=2M

if [[ ${PROFILE} -ge 1 ]]; then
    export TMPDIR="/results/"
fi



HYDRA_LOG_DIR="logs/${CONFIG_FILE%.*}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}"
if [ "${SLURM_LOCALID}" -ne "0" ]; then
    HYDRA_LOG_DIR="$(mktemp -d)"
fi;


GPUS=$(seq 0 $(($NGPUS - 1)) | tr "\n" "," | sed 's/,$//')
PARAMS=(
    -cn ${CONFIG_FILE}

    hydra.run.dir=${HYDRA_LOG_DIR}

    +mpi.local_size=${NGPUS}
    +mpi.local_rank=${local_rank}
    +log.timestamp=${DATESTAMP}
    +log.experiment_id=${EXPERIMENT_ID}

    data.root_dir=${DATAROOT}
)

if [[ ${NUM_INSTANCES} -gt 1 ]]; then
    PARAMS+=(
        +instances=${NUM_INSTANCES}
    )
fi

if [ "${MLPERF_POWER_TRAIN_AFTER_RUN_STOP}" != "1" ]; then
    PARAMS+=(
        +training.early_stop=True
    )
fi

if [ -n "${SLURM_LOCALID-}" ]; then
  # Mode 1: Slurm launched a task for each GPU and set some envvars; nothing to do
  DISTRIBUTED=
else
  # Mode 2: Single-node Docker; need to launch tasks with mpirun
  DISTRIBUTED="mpirun --allow-run-as-root --bind-to none --np ${DGXNGPU}"
fi

PROFILE_COMMAND=""
if [[ ${PROFILE} -ge 1 ]]; then
    if [[ ${global_rank} == 0 ]]; then
        if [[ ${local_rank} == 0 ]] || [[ ${PROFILE_ALL_LOCAL_RANKS} == 1 ]]; then
            PROFILE_COMMAND="nsys profile --trace=cuda,nvtx --force-overwrite true --cuda-graph-trace=node --export=sqlite --output /results/${NETWORK}_b${BATCHSIZE}_%h_${local_rank}_${global_rank}.qdrep "
            PARAMS+=(
                profile=True
            )

            if [[ ${CUDA_PROFILER_RANGE} != "" ]]; then
                PARAMS+=(
                    +profile_range=\"${CUDA_PROFILER_RANGE}\"
                )
                PROFILE_COMMAND="${PROFILE_COMMAND} --capture-range cudaProfilerApi  --capture-range-end stop-shutdown"
            fi
        fi
    fi
fi

IB_BIND=''
if [[ "${SLURM_JOB_NUM_NODES}" -gt 1 && "${ENABLE_IB_BINDING}" -eq 1 ]]; then
  IB_BIND='--ib=single'
fi
BIND="bindpcie --cpu=exclusive ${IB_BIND} --"

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

if [[ ${PROFILE} -ge 1 ]]; then
    TMPDIR=/results ${DISTRIBUTED} ${BIND} ${PROFILE_COMMAND} python main.py "${PARAMS[@]}" "$@"; ret_code=$?
else
    ${LOGGER:-} ${DISTRIBUTED} ${BIND} python main.py "${PARAMS[@]}" "$@"; ret_code=$?
fi

sleep 3

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="COSMOFLOW_HPC"
echo "RESULT,$result_name,,$result,$USER,$start_fmt"
export PROFILE=0
