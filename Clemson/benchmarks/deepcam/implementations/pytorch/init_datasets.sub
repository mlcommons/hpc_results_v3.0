#!/bin/bash
#SBATCH --job-name mlperf-hpc:deepcam

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

# This script assumes that the original HDF5 based dataset was downloaded as described in the README
# file and stored under the directory <hdf5-dataset-path>
# We also assume that the docker container was build with the Dockerfile provided in the docker
# subdirectory and the image was tagged as mlperf-deepcam:v1.0
# Finally, we assume the cluster you are running uses SLURM and pyxis/enroot for running containers.
# The variable RANKS_PER_NODE can be increased to give better throughput
export CONTAINER_IMAGE=mlperf-deepcam:v1.0
export DATASET_INPUT_DIR=<hdf5-dataset-path>
export DATASET_OUTPUT_DIR=<numpy-dataset-path>
export RANKS_PER_NODE=1

# create container mounts
readonly _cont_name="mlperf_deepcam_convert_dataset"
readonly _cont_mounts="${DATASET_INPUT_DIR}:/data_in:ro,${DATASET_OUTPUT_DIR}:/data_out:rw"

# create directories and copy statsfile
mkdir -p ${DATASET_OUTPUT_DIR}/train
mkdir -p ${DATASET_OUTPUT_DIR}/validation
cp ${DATASET_INPUT_DIR}/stats.h5 ${DATASET_OUTPUT_DIR}/

# now, do the conversion:
srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${CONTAINER_IMAGE}" --container-name=${_cont_name} true

# run the conversion for training data
srun --mpi=pmix --ntasks=$(( ${SLURM_JOB_NUM_NODES} * ${RANKS_PER_NODE} )) --ntasks-per-node=${RANKS_PER_NODE} \
     --container-name=${_cont_name} --container-mounts="${_cont_mounts}" \
     --container-workdir /workspace \
     python /opt/utils/convert_hdf52npy.py \
     --input_directory=/data_in/train \
     --output_directory=/data_out/train

# now on validation data
srun --mpi=pmix --ntasks=$(( ${SLURM_JOB_NUM_NODES} * ${RANKS_PER_NODE} )) --ntasks-per-node=${RANKS_PER_NODE} \
     --container-name=${_cont_name} --container-mounts="${_cont_mounts}" \
     --container-workdir /workspace \
     python /opt/utils/convert_hdf52npy.py \
     --input_directory=/data_in/validation \
     --output_directory=/data_out/validation
