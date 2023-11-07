#!/bin/bash
#SBATCH -J deepcam-opt
#SBATCH -A m4291
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --image registry.nersc.gov/das/deepcam-opt:23.09.01

set -euxo pipefail

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

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
#: "${CONT:?CONT not set}"

# Vars with defaults
: "${MLPERF_RULESET:=2.0.0}"
#: "${MLPERF_CLUSTER_NAME:='unknown'}"
#: "${CHECK_COMPLIANCE:=1}"
: "${DGXRUNNODES:=${SLURM_JOB_NUM_NODES}}"
: "${NEXP:=1}"
: "${NUM_INSTANCES:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
#: "${CLEAR_CACHES:=1}"
#: "${LOGDIR:=./results}"
: "${LOGDIR:=$SCRATCH/optimized-hpc/deepcam/results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}}"
#: "${ABSLOGDIR:=${PWD}/results}"
#: "${POWERCMDDIR:=' '}"
#: "${API_LOG_DIR:=./api_logs}" # apiLog.sh output dir
: "${CUDNN_V8_API_ENABLED:=1}"
: "${NCCL_ASYNC_ERROR_HANDLING:=0}"
: "${NCCL_TEST:=1}"
#: "${NCCL_BISECT:=0}"
: "${WIREUP_METHOD:=nccl-slurm}"
: "${ADDITIONAL_SRUN_ARGS:=""}"
#: "${COMP_CLOCK:=${BASE_COMP_CLOCK}}"
#: "${MEM_CLOCK=${BASE_MEM_CLOCK}}"
#: "${DROPCACHE_CMD:="sudo /sbin/sysctl vm.drop_caches=3"}"

# compute number of total ranks
TOTALGPU=$(( ${DGXRUNNODES} * ${DGXNGPU} ))

# determine the wireup method
if [ "${TOTALGPU}" -eq 1 ]; then
    WIREUP_METHOD="dummy"
fi

# pyxis sometimes leaves containers lying around which can really confuse things:
#cleanup_pyxis() {
#    srun --ntasks="${SLURM_JOB_NUM_NODES}" /bin/bash -c 'if [[ "$(enroot list)" ]]; then enroot remove -f $(enroot list); fi'
#}
#trap cleanup_pyxis TERM EXIT
#cleanup_pyxis

# Other vars
export MODEL_NAME="deepcam"
export MODEL_FRAMEWORK="pytorch"
readonly _seed_override=${SEED:-}
readonly _logfile_base="${LOGDIR}/slurm_${DATESTAMP}"
readonly _cont_name="${MODEL_NAME}_${SLURM_JOB_ID}"
#_cont_mounts="${DATADIR}:/data:ro,${LOGDIR}:/results:rw,/raid/scratch:/scratch:rw"
_cont_mounts="${DATADIR}:/data:ro;${LOGDIR}:/results" #;${STAGING_DIR}:/scratch"

#if [ "${API_LOGGING:-0}" -eq 1 ]; then
#    API_LOG_DIR=${API_LOG_DIR}/${MODEL_FRAMEWORK}/${MODEL_NAME}/${DGXSYSTEM}
#    mkdir -p ${API_LOG_DIR}
#    _cont_mounts="${_cont_mounts},${API_LOG_DIR}:/logs"
#fi
#if [ "${JET:-0}" -eq 1 ]; then
#    _cont_mounts="${_cont_mounts},${JET_DIR}:/root/.jet"
#fi
#
#if [ "${ENABLE_GDS:-0}" == "1" ]; then
#    echo "GDS enabled"
#    _cont_mounts="${_cont_mounts},/run/udev:/run/udev:ro"
#fi

#if [ "${SBATCH_NETWORK:-}" == "sharp" ]; then
#    echo "Using SHARP"
#    #export SHARP_COLL_LOCK_ON_COMM_INIT=1
#    #export SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0
#    #export SHARP_COLL_ENABLE_SAT=1
#    #export NCCL_COLLNET_ENABLE=1
#    #export SHARP_COLL_SHARPD_SOCKET_NAME=sharpd_hpcx_2.4.2
#    if [ "${SHARP_DEBUG:-0}" -eq 1 ]; then
#	export SHARP_COLL_LOG_LEVEL=3
#	export NCCL_DEBUG=info
#    fi
#fi

# MLPerf vars
#MLPERF_HOST_OS=$(srun -N1 -n1 bash <<EOF
#		 source /etc/os-release
#		 source /etc/dgx-release || true
#		 echo "\${PRETTY_NAME} / \${DGX_PRETTY_NAME:-???} \${DGX_OTA_VERSION:-\${DGX_SWBUILD_VERSION:-???}}"
#EOF
#)
#export MLPERF_HOST_OS

# Setup directories
#( umask 0002; mkdir -p "${LOGDIR}" )
mkdir -p "${LOGDIR}"

# set clock if reqeusted
#if [ ${COMP_CLOCK} -ne ${BASE_COMP_CLOCK} ] || [ ${MEM_CLOCK} -ne ${BASE_MEM_CLOCK} ]; then
#    if [ ${SLURM_PROCID} -eq 0 ]; then
#	echo "Setting compute clock to ${COMP_CLOCK}"
#    fi
#    srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 bash -c "sudo nvidia-smi -ac ${MEM_CLOCK},${COMP_CLOCK}"
#fi

# Setup container
#srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${CONT}" --container-name="${_cont_name}" true

# print binding info
#srun -N1 -n1 --container-name="${_cont_name}" --no-container-mount-home ibv_devinfo --list
#srun -N1 -n1 --container-name="${_cont_name}" --no-container-mount-home nvidia-smi topo -m

# print python package versions
#srun --ntasks=1 --ntasks-per-node=1 --container-name="${_cont_name}" --no-container-mount-home bash -c "conda list; pip list"

# NCCL Test if requested
#echo "NCCL_TEST = ${NCCL_TEST}"
#if [[ ${NCCL_TEST} -eq 1 ]]; then
#    (srun --mpi="${SLURM_MPI_TYPE:-pmix}" --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
#         --container-name="${_cont_name}" all_reduce_perf_mpi -b 210M -e 220M -d float -G 1 -f 2
#) |& tee "${LOGDIR}/${SPREFIX}_nccl.log"
#fi

# NCCL BISECT test
#if [ ${NCCL_BISECT} -eq 1 ] && [ "${SBATCH_NETWORK}" != "sharp" ]; then
#    ./gpucommtest/gpucommtest.sh --stats --container-name="${_cont_name}"
#fi

# ssh to nodes for power measurements
#NODELIST=$(scontrol show hostnames ${SLURM_JOB_NODELIST})
#NODELIST=(${NODELIST[*]})
#if [ -f "$POWERCMDDIR/power_monitor.sh"  ]; then
#    ( umask 0002; mkdir -p "${ABSLOGDIR}" )
#    for i in "${NODELIST[@]}"
#    do
#        ssh $i 'export NODENAME='"'$i'"';export ABSLOGDIR='"'$ABSLOGDIR'"';export SLURM_JOB_NODELIST='"'$SLURM_JOB_NODELIST'"';export SLURM_JOB_ID='"'$SLURM_JOB_ID'"';POWERCMDDIR='"'$POWERCMDDIR'"';bash ${POWERCMDDIR}/power_monitor.sh' &
##	break
#    done
#fi

# Multi-instance, throughput measurement jobs
if [[ $NUM_INSTANCES -gt 1 ]]; then

    ## Clear caches
    #if [ "${CLEAR_CACHES}" -eq 1 ]; then
    #    srun --ntasks="${SLURM_JOB_NUM_NODES}" --mpi="${SLURM_MPI_TYPE:-pmix}" bash -c "echo -n 'Clearing cache on ' && hostname && sync && ${DROPCACHE_CMD}"
    #fi

    JOB_NODES=$((SLURM_JOB_NUM_NODES / NUM_INSTANCES))
    seed=${_seed_override:-$(date +%s)}

    for _job_index in $(seq 1 "${NUM_INSTANCES}"); do
        export SEED=$((seed + _job_index))
        export EXP_ID=${_job_index}
        export DATESTAMP=${DATESTAMP}
        export WIREUP_METHOD=${WIREUP_METHOD}
        export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING}


	srun --wait=900 --kill-on-bad-exit=0 --mpi=pmi2 \
            --cpus-per-task=32 --cpu-bind=none ${ADDITIONAL_SRUN_ARGS} \
            -N "${JOB_NODES}" \
            --ntasks="$(( JOB_NODES * DGXNGPU ))" \
            --ntasks-per-node="${DGXNGPU}" \
            shifter --volume="${_cont_mounts}" --module gpu,nccl-2.18 \
	    bash ./run_and_time.sh &

        sleep 1
        logging_filename="${LOGDIR}/${DATESTAMP}_${EXP_ID}.log"
        ID=$(sacct -j $SLURM_JOB_ID --format JobID --parsable2 | tail -n 1)
        NNODES=$(sacct -j $SLURM_JOB_ID --format NNodes --parsable2 | tail -n 1)
        NODE_LIST=$(sacct -j $SLURM_JOB_ID --format NodeList%50 --parsable2 | tail -n 1)
        #echo ":::DLPAL ${CONT} ${ID} ${NNODES} ${NODE_LIST} ${MLPERF_CLUSTER_NAME} ${DGXSYSTEM}" >> $logging_filename
    done
    wait

else

    # Single-instance, time-to-train measurement jobs
    for _experiment_index in $(seq 1 "${NEXP}"); do
        (
	echo "Beginning trial ${_experiment_index} of ${NEXP}"
	#echo ":::DLPAL ${CONT} ${SLURM_JOB_ID} ${SLURM_JOB_NUM_NODES} ${SLURM_JOB_NODELIST} ${MLPERF_CLUSTER_NAME} ${DGXSYSTEM}"

	# Clear caches
	#if [ "${CLEAR_CACHES}" -eq 1 ]; then
	#    srun --ntasks="${SLURM_JOB_NUM_NODES}" --mpi="${SLURM_MPI_TYPE:-pmix}" bash -c "echo -n 'Clearing cache on ' && hostname && sync && ${DROPCACHE_CMD}"
	#fi

	# Set Vars
	export SEED=${_seed_override:-$(date +%s)}
	export EXP_ID=${_experiment_index}
	export DATESTAMP=${DATESTAMP}
	export WIREUP_METHOD=${WIREUP_METHOD}
	export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING}

	# Run experiment
	srun --wait=900 --kill-on-bad-exit=0 --mpi=pmi2 \
             --cpus-per-task=32 --cpu-bind=none ${ADDITIONAL_SRUN_ARGS} \
	     -N "${DGXRUNNODES}" \
	     --ntasks="${TOTALGPU}" \
	     --ntasks-per-node="${DGXNGPU}" \
             shifter --volume="${_cont_mounts}" --module gpu,nccl-2.18 \
	     bash ./run_and_time.sh
        ) |& tee "${_logfile_base}_${_experiment_index}.log"

        # compliance checker
        #if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
        #  srun --ntasks=1 --nodes=1 --container-name="${_cont_name}" \
        #       --container-mounts="$(realpath ${LOGDIR}):/results"   \
        #       --container-workdir="/results"                        \
        #       python3 -m mlperf_logging.compliance_checker --usage hpc \
        #       --ruleset "${MLPERF_RULESET}"                                 \
        #       --log_output "/results/compliance_${DATESTAMP}_${_experiment_index}.out"           \
        #       "/results/slurm_${DATESTAMP}_${_experiment_index}.log" \
        # || true
        #fi

        #if [ "${JET:-0}" -eq 1 ]; then
        #  JET_CREATE=${JET_CREATE:-}" --data job_id=${SLURM_JOB_ID} --data pipeline_id=${CONT} --data workload.spec.nodes=${DGXNNODES} --data workload.spec.name=${MODEL_NAME}_${MODEL_FRAMEWORK}_${DGXSYSTEM} --data workload.key=${MODEL_NAME}_${MODEL_FRAMEWORK}_${DGXSYSTEM} --mllogger "
        #  srun -N1 -n1 --container-name="${_cont_name}" --container-mounts="${_cont_mounts}" bash -c "${JET_CREATE} /results/slurm_${DATESTAMP}_${_experiment_index}.log && ${JET_UPLOAD}"
        #fi

    done
    wait
fi

#if [ ${COMP_CLOCK} -ne ${BASE_COMP_CLOCK} ] || [ ${MEM_CLOCK} -ne ${BASE_MEM_CLOCK} ]; then
#    if [ ${SLURM_PROCID} -eq 0 ]; then
#        echo "Resetting compute clock to ${BASE_COMP_CLOCK}"
#    fi
#    srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 bash -c "sudo nvidia-smi -ac ${BASE_MEM_CLOCK},${BASE_COMP_CLOCK}"
#fi
#wait
