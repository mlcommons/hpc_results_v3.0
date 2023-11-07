#!/bin/bash
#SBATCH -J mlperf-openfold-optimized
#SBATCH -A m4291
#SBATCH -C gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --image=sfarrell/openfold-ref:23.02
#SBATCH --module=gpu,nccl-2.15
#SBATCH --output=openfold_sterr_shifter_pq_8.log

# Vars with defaults
NEXP="${NEXP:-1}"
export LOGDIR="${LOGDIR:-${SCRATCH}/openfold-ref/results/${SLURM_JOB_ID}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/openfold-ref/profile/optimized-pq-${SLURM_JOB_ID}}"
export DATA_DIR="${DATA_DIR:-${SCRATCH}/openfold/pdb_data}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SCRATCH}/openfold/mlperf_hpc_openfold_resumable_checkpoint_v2.pt}"

# Other settings
export MASTER_ADDR=$(hostname)
export FI_MR_CACHE_MONITOR=userfaultfd
export OMP_NUM_THREADS=1
# 12 dataloader workers work best
# if multiprocess multithreaded execution is needed, 
# OMP_NUM_THREADS can be increased such that 
# num_threads * (sampler worker + dataloader workers) < SLURM_CPUS_PER_TASK

#export LD_LIBRARY_PATH=/global/common/software/nersc/pm-2023q1/sw/nccl-2.17.1-ofi-cuda11/lib:${LD_LIBRARY_PATH}

# Extra command line args
args=$@

# Setup directories
mkdir -p "${LOGDIR}"
mkdir -p "${OUTPUT_DIR}"

# Run experiments
for iexp in $(seq 1 "${NEXP}"); do

    echo "Beginning trial ${iexp} of ${NEXP}"

    # Run experiment
    #export SEED=${_seed_override:-$RANDOM}
    srun -u --mpi=pmi2 shifter \
        bash scripts/run_training.sh  ${args}

done

