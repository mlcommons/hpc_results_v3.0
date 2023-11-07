#!/bin/bash
#SBATCH -J opencatalyst-opt
#SBATCH -A m4291
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node 4
#SBATCH --image registry.nersc.gov/das/sfarrell/opencatalyst-opt:23.09.01

###SBATCH --image registry.nersc.gov/das/opencatalyst-opt:23.09.00

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


set -euxo pipefail

# Vars without defaults
#: "${CONT:?CONT not set}"

# Vars with defaults
: "${MLPERF_RULESET:=2.0.0}"
#: "${MLPERF_CLUSTER_NAME:='unknown'}"
#: "${CHECK_COMPLIANCE:=1}"
#: "${DGXSYSTEM:=DGXA100}"
: "${DGXNGPU:=4}"
: "${NEXP:=1}"
: "${NCCL_TEST:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
#: "${CLEAR_CACHES:=1}"
#: "${LOGDIR:=./results}"
: "${LOGDIR:=$SCRATCH/optimized-hpc/opencatalyst/results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}}"
#: "${ABSLOGDIR:=${PWD}/results}"
#: "${POWERCMDDIR:=' '}"
#: "${API_LOG_DIR:=./api_logs}" # apiLog.sh output dir
#: "${DROPCACHE_CMD:="sudo /sbin/sysctl vm.drop_caches=3"}"

# pyxis sometimes leaves containers lying around which can really confuse things:
#cleanup_pyxis() {
#    srun --ntasks="${SLURM_JOB_NUM_NODES}" /bin/bash -c 'if [[ "$(enroot list)" ]]; then enroot remove -f $(enroot list); fi'
#}
#trap cleanup_pyxis TERM EXIT
#cleanup_pyxis


# Other vars
export MODEL_NAME="oc20"
export MODEL_FRAMEWORK="pytorch"
readonly _seed_override=${SEED:-}
readonly _logfile_base="${LOGDIR}/slurm_${DATESTAMP}"
readonly _cont_name="${MODEL_NAME}_${SLURM_JOB_ID}"
_cont_mounts="${DATADIR}:/data:ro;${LOGDIR}:/results"
#SPREFIX="${MODEL_NAME}_${MODEL_FRAMEWORK}_${DGXNNODES}x${DGXNGPU}x${BATCH_SIZE}_${DATESTAMP}"

#if [ "${API_LOGGING:-0}" -eq 1 ]; then
#    API_LOG_DIR=${API_LOG_DIR}/${MODEL_FRAMEWORK}/${MODEL_NAME}/${DGXSYSTEM}
#    mkdir -p ${API_LOG_DIR}
#    _cont_mounts="${_cont_mounts},${API_LOG_DIR}:/logs"
#fi
#if [ "${JET:-0}" -eq 1 ]; then
#    _cont_mounts="${_cont_mounts},${JET_DIR}:/root/.jet"
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
#srun --ntasks="${SLURM_JOB_NUM_NODES}" mkdir -p "${LOGDIR}"
mkdir -p "${LOGDIR}"

# Setup container
#srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${CONT}" --container-name="${_cont_name}" true

#echo "NCCL_TEST = ${NCCL_TEST}"
#if [[ ${NCCL_TEST} -eq 1 ]]; then
#    (srun --mpi="${SLURM_MPI_TYPE:-pmix}" --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
#         --container-name="${_cont_name}" all_reduce_perf_mpi -b 3542K -e 3542K -d half -G 1 -f 2
#) |& tee "${LOGDIR}/${SPREFIX}_nccl.log"
#fi


## ssh to nodes for power measurements
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

# Run experiments
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

	# Run experiment
	srun -l --kill-on-bad-exit=1 --mpi=pmi2 \
            --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" \
            --ntasks-per-node="${DGXNGPU}" \
            --cpus-per-task=32 --cpu-bind=none \
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
