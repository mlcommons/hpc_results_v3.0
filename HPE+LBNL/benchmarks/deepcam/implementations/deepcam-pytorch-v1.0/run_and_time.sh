#!/bin/bash
#
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# assemble launch command
export TOTALGPUS=$(( ${SLURM_NNODES} * ${DGXNGPU} ))
if [ ! -z ${TRAINING_INSTANCE_SIZE} ]; then
    gpu_config="$(( ${TOTALGPUS} / ${TRAINING_INSTANCE_SIZE} ))x${TRAINING_INSTANCE_SIZE}"
else
    gpu_config=${TOTALGPUS}
fi
export RUN_TAG=${RUN_TAG:-run_${DATESTAMP}}
export OUTPUT_DIR=${OUTPUT_ROOT:-/results}

# Set PMIX variable assumed by the deepcam source code
export PMIX_RANK=${SLURM_PROCID}

# Dumping slurm environment from rank 0
if [[ "${SLURM_PROCID}" == "0" ]]; then
    env | grep SLURM
fi

# create tmp directory
mkdir -p ${OUTPUT_DIR}

# LR switch
if [ -z ${LR_SCHEDULE_TYPE} ]; then
    lr_schedule_arg=""
else
    lr_schedule_arg="--lr_schedule type=\"${LR_SCHEDULE_TYPE}\",milestones=\"${LR_MILESTONES}\",decay_rate=\"${LR_DECAY_RATE}\""
fi

PARAMS=(
    --wireup_method ${WIREUP_METHOD}
    --run_tag ${RUN_TAG}
    --experiment_id ${EXP_ID:-1}
    --data_dir_prefix ${DATA_DIR_PREFIX:-"/data"}
    --output_dir ${OUTPUT_DIR}
    --model_prefix "segmentation"
    --optimizer ${OPTIMIZER}
    --start_lr ${START_LR}
    --lr_schedule type="${LR_SCHEDULE_TYPE}",milestones="${LR_MILESTONES}",decay_rate="${LR_DECAY_RATE}"
    --lr_warmup_steps ${LR_WARMUP_STEPS}
    --lr_warmup_factor ${LR_WARMUP_FACTOR}
    --weight_decay ${WEIGHT_DECAY}
    --logging_frequency ${LOGGING_FREQUENCY}
    --save_frequency 100000
    --max_epochs ${MAX_EPOCHS:-200}
    --max_inter_threads ${MAX_THREADS:-4}
    --seed ${SEED}
    --batchnorm_group_size ${BATCHNORM_GROUP_SIZE}
    --shuffle_mode "${SHUFFLE_MODE}"
    --data_format "${DATA_FORMAT}"
    --data_oversampling_factor ${DATA_OVERSAMPLING_FACTOR:-1}
    --precision_mode "${PRECISION_MODE}"
    --enable_nhwc
    --local_batch_size ${LOCAL_BATCH_SIZE}
    --local_batch_size_validation ${LOCAL_VALIDATION_BATCH_SIZE}
    ${ADDITIONAL_ARGS}
)

# change directory
pushd ./deepCam

# profile command:
if [ ! -z ${OMPI_COMM_WORLD_RANK} ]; then
    WORLD_RANK=${OMPI_COMM_WORLD_RANK}
elif [ ! -z ${PMIX_RANK} ]; then
    WORLD_RANK=${PMIX_RANK}
elif [ ! -z ${PMI_RANK} ]; then
    WORLD_RANK=${PMIXRANK}
fi
PROFILE_BASE_CMD="nsys profile --mpi-impl=openmpi --trace=cuda,cublas,nvtx,mpi -f true -o ${OUTPUT_DIR}/profile_rank${WORLD_RANK}"
if [[ ${ENABLE_PROFILING} == 1 ]]; then
    echo "Profiling enabled"
    PROFILE_CMD=${PROFILE_BASE_CMD}
else
    PROFILE_CMD=""
fi

if [[ ${DEBUG_MEMCHECK} == 1 ]]; then
    echo "Debugging enabled"
    DEBUG_CMD="compute-sanitizer --tool=memcheck"
else
    DEBUG_CMD=""
fi

# bind command
if [[ "${NERSC_HOST}" == "perlmutter" ]]; then
    BIND_CMD="${SCRIPT_DIR}/bind.sh --cpu=${SCRIPT_DIR}/pm_bind_map.sh --mem=${SCRIPT_DIR}/pm_bind_map.sh"
else
    BIND_CMD="${SCRIPT_DIR}/bind.sh --cpu=exclusive"
fi

# do we cache data
if [ ! -z ${DATA_CACHE_DIRECTORY} ]; then
    PARAMS+=(--data_cache_directory ${DATA_CACHE_DIRECTORY})
fi

# run script selection:
if [ ! -z ${TRAINING_INSTANCE_SIZE} ]; then
    echo "Running Multi Instance Training"
    RUN_SCRIPT="./train_instance_oo.py"
    PARAMS+=(--training_instance_size ${TRAINING_INSTANCE_SIZE})

    if [ ! -z ${STAGE_DIR_PREFIX} ]; then
	PARAMS+=(
	    --stage_dir_prefix ${STAGE_DIR_PREFIX}
	    --stage_num_workers ${STAGE_NUM_WORKERS:-1}
	    --stage_batch_size ${STAGE_BATCH_SIZE:--1}
	    --stage_mode ${STAGE_MODE:-"node"}
	)
	# do we need to verify the staging results
	if [ "${STAGE_VERIFY:-0}" -eq 1 ]; then
	    PARAMS+=(--stage_verify)
	fi
	if [ "${STAGE_ONLY:-0}" -eq 1 ]; then
	    echo "WARNING: You are about to run a staging only benchmark"
	    PARAMS+=(--stage_only)
	fi
	if [ "${STAGE_FULL_DATA_PER_NODE:-0}" -eq 1 ]; then
	    PARAMS+=(--stage_full_data_per_node)
	fi
	if [ "${STAGE_ARCHIVES:-0}" -eq 1 ]; then
	    PARAMS+=(--stage_archives)
	fi
	if [ "${STAGE_USE_DIRECT_IO:-0}" -eq 1 ]; then
	    PARAMS+=(--stage_use_direct_io)
	fi 
    fi
elif [ ! -z ${CAPTURE_RANGE_START} ]; then
    echo "Running Profiling"
    RUN_SCRIPT="./profile.py"
    PARAMS+=(
	--capture_range_start ${CAPTURE_RANGE_START}
	--capture_range_stop ${CAPTURE_RANGE_STOP}
    ) 
else
    echo "Running Single Instance Training"
    RUN_SCRIPT="./train.py"
fi

# run command
${BIND_CMD} ${PROFILE_CMD} ${DEBUG_CMD} python ${RUN_SCRIPT} "${PARAMS[@]}"; ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="DEEPCAM_HPC"
echo "RESULT,$result_name,,$result,$USER,$start_fmt"
