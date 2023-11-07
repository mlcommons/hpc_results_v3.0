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

set -e  # immediately exit on first error

# Setup text effects:
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
BOLD=$(tput bold)
NORMAL=$(tput sgr0)

set -x  # print commands

DATETIME=$(date +'%y%m%d_%H%M%S')

# set DGXSYSTEM variable:
GPU=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0 | grep -oE "A100|H100" | head -1)
export DGXSYSTEM="DGX${GPU}"

mkdir -p /results/$DATETIME/

RUNCMD=${1-single}

TRAINING_DIRPATH=/results/$DATETIME
PDB_MMCIF_CHAINS_FILEPATH=/data/pdb_mmcif/processed/chains.csv
PDB_MMCIF_DICTS_DIRPATH=/data/pdb_mmcif/processed/dicts
PDB_OBSOLETE_FILEPATH=/data/pdb_mmcif/processed/obsolete.dat
PDB_ALIGNMENTS_DIRPATH=/data/open_protein_set/processed/pdb_alignments
INITIALIZE_PARAMETERS_FROM=/data/mlperf_hpc_openfold_resumable_checkpoint.pt
PRECISION="bf16"
SEED=1234567890
NUM_TRAIN_ITERS=40
LOG_EVERY_ITERS=1
VAL_EVERY_ITERS=8
LOCAL_BATCH_SIZE=1
NUM_TRAIN_DATALOADER_WORKERS=2
NUM_VAL_DATALOADER_WORKERS=1

if [ "$RUNCMD" == "single" ]; then

echo "${BOLD}${GREEN}${RUNCMD}${NORMAL}"

python train.py \
--training_dirpath $TRAINING_DIRPATH \
--pdb_mmcif_chains_filepath $PDB_MMCIF_CHAINS_FILEPATH \
--pdb_mmcif_dicts_dirpath $PDB_MMCIF_DICTS_DIRPATH \
--pdb_obsolete_filepath $PDB_OBSOLETE_FILEPATH \
--pdb_alignments_dirpath $PDB_ALIGNMENTS_DIRPATH \
--initialize_parameters_from $INITIALIZE_PARAMETERS_FROM \
--train_max_pdb_release_date "2021-12-11" \
--target_avg_lddt_ca_value 0.99 \
--precision $PRECISION \
--seed $SEED \
--num_train_iters $NUM_TRAIN_ITERS \
--log_every_iters $LOG_EVERY_ITERS \
--val_every_iters $VAL_EVERY_ITERS \
--local_batch_size $LOCAL_BATCH_SIZE \
--dap_size 0 \
--num_train_dataloader_workers $NUM_TRAIN_DATALOADER_WORKERS \
--num_val_dataloader_workers $NUM_VAL_DATALOADER_WORKERS \
--use_only_pdb_chain_ids 7ny6_A 7e6g_A \
--disable_warmup

elif [ "$RUNCMD" == "ddp" ]; then

echo "${BOLD}${GREEN}${RUNCMD}${NORMAL}"

NPROC_PER_NODE=${2-2}

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE train.py \
--training_dirpath $TRAINING_DIRPATH \
--pdb_mmcif_chains_filepath $PDB_MMCIF_CHAINS_FILEPATH \
--pdb_mmcif_dicts_dirpath $PDB_MMCIF_DICTS_DIRPATH \
--pdb_obsolete_filepath $PDB_OBSOLETE_FILEPATH \
--pdb_alignments_dirpath $PDB_ALIGNMENTS_DIRPATH \
--initialize_parameters_from $INITIALIZE_PARAMETERS_FROM \
--train_max_pdb_release_date "2021-12-11" \
--target_avg_lddt_ca_value 0.99 \
--precision $PRECISION \
--seed $SEED \
--num_train_iters $NUM_TRAIN_ITERS \
--log_every_iters $LOG_EVERY_ITERS \
--val_every_iters $VAL_EVERY_ITERS \
--local_batch_size $LOCAL_BATCH_SIZE \
--dap_size 0 \
--initial_training_dataloader_type InitialTrainingDataloaderPQ \
--num_train_dataloader_workers $NUM_TRAIN_DATALOADER_WORKERS \
--num_val_dataloader_workers $NUM_VAL_DATALOADER_WORKERS \
--train_dataloader_threading \
--use_only_pdb_chain_ids 7ny6_A 7e6g_A \
--disable_warmup \
--distributed
# | tee /results/$DATETIME/stdout

elif [ "$RUNCMD" == "dap" ]; then

echo "${BOLD}${GREEN}${RUNCMD}${NORMAL}"

DAP_SIZE=${2-2}
NPROC_PER_NODE=${3-4}

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE train.py \
--training_dirpath $TRAINING_DIRPATH \
--pdb_mmcif_chains_filepath $PDB_MMCIF_CHAINS_FILEPATH \
--pdb_mmcif_dicts_dirpath $PDB_MMCIF_DICTS_DIRPATH \
--pdb_obsolete_filepath $PDB_OBSOLETE_FILEPATH \
--pdb_alignments_dirpath $PDB_ALIGNMENTS_DIRPATH \
--initialize_parameters_from $INITIALIZE_PARAMETERS_FROM \
--train_max_pdb_release_date "2021-12-11" \
--target_avg_lddt_ca_value 0.99 \
--precision $PRECISION \
--seed $SEED \
--num_train_iters $NUM_TRAIN_ITERS \
--log_every_iters $LOG_EVERY_ITERS \
--val_every_iters $VAL_EVERY_ITERS \
--local_batch_size $LOCAL_BATCH_SIZE \
--dap_size $DAP_SIZE \
--initial_training_dataloader_type InitialTrainingDataloaderPQ \
--num_train_dataloader_workers $NUM_TRAIN_DATALOADER_WORKERS \
--num_val_dataloader_workers $NUM_VAL_DATALOADER_WORKERS \
--train_dataloader_threading \
--use_only_pdb_chain_ids 7ny6_A 7e6g_A \
--disable_warmup \
--distributed
# | tee /results/$DATETIME/stdout

elif [ "$RUNCMD" == "aval" ]; then

echo "${BOLD}${GREEN}${RUNCMD}${NORMAL}"

NUM_ASYNC_VAL_RANKS=${2-2}
NPROC_PER_NODE=${3-5}

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE train.py \
--training_dirpath $TRAINING_DIRPATH \
--pdb_mmcif_chains_filepath $PDB_MMCIF_CHAINS_FILEPATH \
--pdb_mmcif_dicts_dirpath $PDB_MMCIF_DICTS_DIRPATH \
--pdb_obsolete_filepath $PDB_OBSOLETE_FILEPATH \
--pdb_alignments_dirpath $PDB_ALIGNMENTS_DIRPATH \
--initialize_parameters_from $INITIALIZE_PARAMETERS_FROM \
--train_max_pdb_release_date "2021-12-11" \
--target_avg_lddt_ca_value 0.99 \
--precision $PRECISION \
--seed $SEED \
--num_train_iters $NUM_TRAIN_ITERS \
--log_every_iters $LOG_EVERY_ITERS \
--val_every_iters $VAL_EVERY_ITERS \
--local_batch_size $LOCAL_BATCH_SIZE \
--dap_size 0 \
--initial_training_dataloader_type InitialTrainingDataloaderPQ \
--num_train_dataloader_workers $NUM_TRAIN_DATALOADER_WORKERS \
--num_val_dataloader_workers $NUM_VAL_DATALOADER_WORKERS \
--train_dataloader_threading \
--num_async_val_ranks $NUM_ASYNC_VAL_RANKS \
--use_only_pdb_chain_ids 7ny6_A 7e6g_A 7s3u_A \
--disable_warmup \
--distributed
# | tee /results/$DATETIME/stdout

elif [ "$RUNCMD" == "dap+aval" ]; then

echo "${BOLD}${GREEN}${RUNCMD}${NORMAL}"

DAP_SIZE=${2-2}
NUM_ASYNC_VAL_RANKS=${3-2}
NPROC_PER_NODE=${4-8}

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE train.py \
--training_dirpath $TRAINING_DIRPATH \
--pdb_mmcif_chains_filepath $PDB_MMCIF_CHAINS_FILEPATH \
--pdb_mmcif_dicts_dirpath $PDB_MMCIF_DICTS_DIRPATH \
--pdb_obsolete_filepath $PDB_OBSOLETE_FILEPATH \
--pdb_alignments_dirpath $PDB_ALIGNMENTS_DIRPATH \
--initialize_parameters_from $INITIALIZE_PARAMETERS_FROM \
--train_max_pdb_release_date "2021-12-11" \
--target_avg_lddt_ca_value 0.99 \
--precision $PRECISION \
--seed $SEED \
--num_train_iters $NUM_TRAIN_ITERS \
--log_every_iters $LOG_EVERY_ITERS \
--val_every_iters $VAL_EVERY_ITERS \
--local_batch_size $LOCAL_BATCH_SIZE \
--dap_size $DAP_SIZE \
--initial_training_dataloader_type InitialTrainingDataloaderPQ \
--num_train_dataloader_workers $NUM_TRAIN_DATALOADER_WORKERS \
--num_val_dataloader_workers $NUM_VAL_DATALOADER_WORKERS \
--train_dataloader_threading \
--num_async_val_ranks $NUM_ASYNC_VAL_RANKS \
--use_only_pdb_chain_ids 7ny6_A 7e6g_A 7s3u_A \
--disable_warmup \
--distributed
# | tee /results/$DATETIME/stdout

else

echo "${BOLD}unknown ${RED}${RUNCMD}${NORMAL}"

fi
