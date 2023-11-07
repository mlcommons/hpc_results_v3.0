#!/bin/bash

# get num gpu:
NUM_GPU=$(cat ./configs/config_DGXA100_common.sh | grep DGXNGPU | awk '{split($2,a,"="); print a[2]}')

for config in $(ls ./configs | grep -v config_DGXA100_common | grep -v config_data); do
    # extract tag:
    TAG=$(echo ${config} | awk '{split($1,a,"DGXA100_"); split(a[2],b,".sh"); print(b[1])}')

    # extract scaling type from tag:
    if [[ "${TAG}" =~ .*"weak".* ]]; then
	SCALING_TYPE="weak"
    else
	SCALING_TYPE="strong"
    fi
    
    # get remaining info
    NUM_NODES=$(cat ./configs/${config} | grep DGXNNODES | awk '{split($2,a,"="); print a[2]}')
    LOCAL_BATCH_SIZE=$(cat ./configs/${config} | grep LOCAL_BATCH_SIZE | awk '{split($2,a,"="); print a[2]}')

    # compute global batch size:
    GLOBAL_BATCH_SIZE=$(( ${LOCAL_BATCH_SIZE} * ${NUM_NODES} * ${NUM_GPU} ))
    if [[ "${SCALING_TYPE}" == "strong" ]]; then
	GLOBAL_BATCH_SIZE=$(( ${LOCAL_BATCH_SIZE} * ${NUM_NODES} * ${NUM_GPU} ))
	NODE_TAG="${NUM_NODES}"
    else
	TRAINING_INSTANCE_SIZE=$(cat ./configs/${config} | grep TRAINING_INSTANCE_SIZE | awk '{split($2,a,"="); print a[2]}')
	GLOBAL_BATCH_SIZE=$(( ${LOCAL_BATCH_SIZE} * ${TRAINING_INSTANCE_SIZE} ))
	NUM_INSTANCES=$(( ${NUM_NODES} * ${NUM_GPU} / ${TRAINING_INSTANCE_SIZE} ))
	NUM_NODES_PER_INSTANCE=$(( ${TRAINING_INSTANCE_SIZE} / ${NUM_GPU} ))
	NODE_TAG="${NUM_INSTANCES} x ${NUM_NODES_PER_INSTANCE}"
    fi

    # set shape
    JOB_SHAPE=$(echo ${TAG} | awk '{split($1,a,"_"); print(a[1])}')

    echo $config $NUM_NODES $GLOBAL_BATCH_SIZE $SCALING_TYPE $JOB_SHAPE
    
    # create readme for specific config:
    cp README_template.md configs/README_${TAG}.md
    sed -i "s|JOB_SHAPE|${JOB_SHAPE}|g" configs/README_${TAG}.md
    sed -i "s|SCALING_TYPE|${SCALING_TYPE}|g" configs/README_${TAG}.md
    sed -i "s|NUM_NODES|${NODE_TAG}|g" configs/README_${TAG}.md
    sed -i "s|NUM_GPU|${NUM_GPU}|g" configs/README_${TAG}.md
    sed -i "s|GLOBAL_BATCH_SIZE|${GLOBAL_BATCH_SIZE}|g" configs/README_${TAG}.md
    sed -i "s|CONFIG_FILE|${config}|g" configs/README_${TAG}.md
    cat configs/README_${TAG}.md

    # remove the readme for now to not clutter the directory
    rm configs/README_${TAG}.md
done
