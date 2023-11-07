#!/bin/bash

# Forward additional command line arguments
OTHER_ARGS=$@

# Distributed settings from SLURM
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export GROUP_RANK=$SLURM_NODEID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
export MASTER_PORT=29500
export SCRIPT_DIR=/pscratch/sd/s/schheda/final/optimized-hpc/openfold/pytorch


export TRAIN_MAX_PDB_RELEASE_DATE="2021-09-16"
export VAL_MIN_CAMEO_SUBMISSION_DATE="2021-09-17"
export VAL_MAX_CAMEO_SUBMISSION_DATE="2021-12-11"


#export LD_LIBRARY_PATH=/opt/udiImage/modules/gpu/lib64:${LD_LIBRARY_PATH}
#export LD_LIBRARY_PATH=/global/common/software/nersc/pm-2023q1/sw/nccl-2.17.1-ofi-cuda11/lib:${LD_LIBRARY_PATH}
PROFILE_CMD="nsys profile   --trace=cuda,nvtx --cpuctxsw=none -f true -o ${OUTPUT_DIR}/profile_rank_${RANK}"
set -eux

export TRITON_CACHE_DIR="/dev/shm/triton/job-${SLURM_JOBID}/rank${SLURM_PROCID}"

# bind command
if [[ "${NERSC_HOST}" == "perlmutter" ]]; then
    BIND="${SCRIPT_DIR}/bind.sh --cpu=${SCRIPT_DIR}/pm_bind_map.sh --mem=${SCRIPT_DIR}/pm_bind_map.sh"
else
    BIND="${SCRIPT_DIR}/bind.sh --cpu=exclusive"
fi

export DGXSYSTEM="DGXA100"
# export NCCL_DEBUG=INFO
#### SHS patch fixes from steve, peter
# export LD_LIBRARY_PATH=/pscratch/sd/p/pharring/shspatch/libshs22_Aug7debug:$LD_LIBRARY_PATH
# export FI_CXI_COMPAT=2
# export FI_CXI_RX_MATCH_MODE=software
# export FI_CXI_RDZV_PROTO=alt_read
# Make output directory if necessary
mkdir -p ${LOGDIR}
mkdir -p ${OUTPUT_DIR}

${BIND} python3 -u train.py \
    --training_dirpath ${LOGDIR} \
    --pdb_mmcif_chains_filepath ${DATA_DIR}/pdb_mmcif/processed/chains.csv \
    --pdb_mmcif_dicts_dirpath ${DATA_DIR}/pdb_mmcif/processed/dicts \
    --pdb_obsolete_filepath ${DATA_DIR}/pdb_mmcif/processed/obsolete.dat \
    --pdb_alignments_dirpath ${DATA_DIR}/open_protein_set/processed/pdb_alignments \
    --initialize_parameters_from ${CHECKPOINT_PATH} \
    --train_max_pdb_release_date ${TRAIN_MAX_PDB_RELEASE_DATE} \
    --val_min_cameo_submission_date ${VAL_MIN_CAMEO_SUBMISSION_DATE} \
    --val_max_cameo_submission_date ${VAL_MAX_CAMEO_SUBMISSION_DATE} \
    --initial_training_dataloader_type InitialTrainingDataloaderPQ \
    --seed ${SEED:-28111995} \
    --num_train_iters ${NUM_TRAIN_ITERS:-2000} \
    --val_every_iters 40 \
    --local_batch_size 1 \
    --base_lr 1e-3 \
    --dap_size 4 \
    --precision bf16 \
    --warmup_lr_init 1e-5 \
    --warmup_lr_iters 0 \
    --num_train_dataloader_workers 14 \
    --num_val_dataloader_workers 4 \
    --num_async_val_ranks 24 \
    ${OTHER_ARGS}                    
