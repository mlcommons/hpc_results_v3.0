#!/bin/bash
#SBATCH --job-name mlperf-hpc:deepcam
#SBATCH -N 18                     # number of nodes
#SBATCH -n 36                     # total number of processes
#SBATCH --ntasks-per-node 2      # tasks per node
#SBATCH -t 12:00:00             # wall time
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --gres=gpu:2

module purge
#module load openmpi/4.1.5
#module load intel-oneapi-mpi/2021.7.1
#module avail
#module load shared
#module load slurm
#module load shared openmpi/4.1.1
#module load shared openmpi/4.1.2

#module list

set -euxo pipefail

#set -x
# The MIT License (MIT)
#
# Copyright (c) 2020-2022 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 


#source config_32xXE8545x4A100-SXM4-40GB.sh

GPU_TYPE=$( nvidia-smi -L | awk '{ print $2 }'| head -n 1 )

source configs/config_18x2A100-80GB.sh
echo source configs/config_18x2A100-80GB.sh
export UCX_POSIX_USE_PROC_LINK=n

CONT="/scratch/nnisbet/mlperf_hpc-deepcam_latest.sif"
#CONT="./mlperf_hpc-deepcam_latest.sif"
: "${LOGDIR:=/project/rcde/nnisbet/mlperf_hpc/submissions/output_deepcam}"
: "${NEXP:=10}"
#: "${DATADIR:=/lustre/fsw/mlperf/mlperf-hpc/tkurth/deepcam_v0.7/data/All-Hist/numpy}"
: "${DATADIR:=/project/rcde/mlperf_data/deepcam/deepcam-numpy}"

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"

# Vars with defaults
: "${DGXRUNNODES:=${SLURM_JOB_NUM_NODES}}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${API_LOG_DIR:=./api_logs}" # apiLog.sh output dir
: "${CUDNN_V8_API_ENABLED:=1}"
: "${NCCL_ASYNC_ERROR_HANDLING:=0}"
: "${NCCL_TEST:=0}"
: "${NCCL_BISECT:=0}"
: "${WIREUP_METHOD:=nccl-slurm}"

# compute number of total ranks
TOTALGPU=$(( ${DGXRUNNODES} * ${DGXNGPU} ))

# determine the wireup method
if [ "${TOTALGPU}" -eq 1 ]; then
    WIREUP_METHOD="dummy"
fi

# Other vars
readonly _seed_override=${SEED:-}
readonly _logfile_base="${LOGDIR}/slurm_${DATESTAMP}"
readonly _cont_name=mlperf-hpc-deepcam
_cont_mounts="${DATADIR}:/data:ro,${LOGDIR}:/results:rw,/scratch/$USER:/scratch:rw"
if [ "${API_LOGGING:-0}" -eq 1 ]; then
    _cont_mounts="${_cont_mounts},${API_LOG_DIR}:/logs"
fi

if [ "${ENABLE_GDS:-0}" == "1" ]; then
	echo "skipping ENABLE_GDS"
    	#_cont_mounts="${_cont_mounts},/run/udev:/run/udev:ro"
fi

if [ "${SBATCH_NETWORK:-}" == "sharp" ]; then
    echo "Using SHARP"
    #export SHARP_COLL_LOCK_ON_COMM_INIT=1
    #export SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0
    #export SHARP_COLL_ENABLE_SAT=1
    #export NCCL_COLLNET_ENABLE=1
    #export SHARP_COLL_SHARPD_SOCKET_NAME=sharpd_hpcx_2.4.2
    if [ "${SHARP_DEBUG:-0}" -eq 1 ]; then
	export SHARP_COLL_LOG_LEVEL=3
	export NCCL_DEBUG=info
    fi
fi

# MLPerf vars
MLPERF_HOST_OS=$(srun -N1 -n1 bash <<EOF
		 source /etc/os-release
		 source /etc/dgx-release || true
		 echo "\${PRETTY_NAME} / \${DGX_PRETTY_NAME:-???} \${DGX_OTA_VERSION:-\${DGX_SWBUILD_VERSION:-???}}"
EOF
)
export MLPERF_HOST_OS

# Setup directories
( umask 0002; mkdir -p "${LOGDIR}" )

## Setup container
#srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${CONT}" --container-name="${_cont_name}" true

## print python package versions
#srun --ntasks=1 --ntasks-per-node=1 --container-name="${_cont_name}" --no-container-mount-home bash -c "conda list; pip list"

# NCCL Test if requested
if [ ${NCCL_TEST} -eq 1 ]; then
    srun  --mpi=pmi2 \
	 -N "${DGXRUNNODES}" \
	 --ntasks="${TOTALGPU}" \
	 --ntasks-per-node="${DGXNGPU}" \
	 --container-name="${_cont_name}" all_reduce_perf_mpi -b 21M -e 270M -d float -G 1 -f 2
fi

# NCCL BISECT test
if [ ${NCCL_BISECT} -eq 1 ] && [ "${SBATCH_NETWORK}" != "sharp" ]; then
    ./gpucommtest/gpucommtest.sh --stats --container-name="${_cont_name}"
fi

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (
	echo "Beginning trial ${_experiment_index} of ${NEXP}"
	
	hosts=$(scontrol show hostname |tr "\n" " ")
        echo "hosts=$hosts"
        #for node_id in `seq 0 $(($NUM_NODES-1))`; do
        for node in $hosts; do

	# Clear caches
	if [ "${CLEAR_CACHES}" -eq 1 ]; then
	    srun --ntasks="${SLURM_JOB_NUM_NODES}" bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
	fi
	done

	# Set Vars
	export SEED=${_seed_override:-$(date +%s)}
	export EXP_ID=${_experiment_index}
	export DATESTAMP=${DATESTAMP}
	export WIREUP_METHOD=${WIREUP_METHOD}
	export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING}

	# Run experiment

#	srun -l --wait=900 --kill-on-bad-exit=0 --mpi=pmix \
#	     -N "${DGXRUNNODES}" \
#	     --ntasks="${TOTALGPU}" \
#	     --ntasks-per-node="${DGXNGPU}" \
#	     --no-container-mount-home \
#	     --container-name="${_cont_name}" --container-mounts="${_cont_mounts}" \
#	     --container-workdir /workspace \
#	     bash ./run_and_time.sh
#    ) |& tee "${_logfile_base}_${_experiment_index}.out"


CONT_MOUNTS="${_cont_mounts}"
echo CONT_MOUNTS="${_cont_mounts}"

        #srun -l --wait=900 --kill-on-bad-exit=0 --ntasks=${SLURM_NTASKS}  --ntasks-per-node=${DGXNGPU} \
	srun -l --overlap --mpi=pmi2 --wait=900 --kill-on-bad-exit=0 --ntasks=${SLURM_NTASKS}  --ntasks-per-node=${DGXNGPU} \
           apptainer exec --nv -B "${CONT_MOUNTS}" \
             -B $PWD:/workspace --pwd /workspace \
             $CONT bash ./run_and_time_multi.sh
    ) |& tee "${_logfile_base}_${_experiment_index}.out"

done
wait
