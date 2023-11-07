#!/bin/bash
#SBATCH -J mlperf-openfold-optimized
#SBATCH -A dasrepo
#SBATCH -C gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=/global/homes/s/schheda/my_repo/optimized-hpc/openfold_sterr_baremetal_pq_21.log

# Vars with defaults

module unload cray-dsmml cray-libsci perftools-base
module unload Nsight-Compute Nsight-Systems xalt

module load cudnn/8.9.1_cuda11 nccl/2.15.5-ofi
#module load cray-hdf5-parallel/1.12.2.3
#module swap gcc/11.2.0 gcc/12.2.0 

#module use ~/modulefiles
#module load tcmalloc

export PREFIX=/mscratch/sd/s/schheda/sw-env
export LD_LIBRARY_PATH=${PREFIX}/opt21/lib:${LD_LIBRARY_PATH}
export PATH=${PREFIX}/opt21/bin:${PATH}:/sbin
source ${PREFIX}/env21/bin/activate

NEXP="${NEXP:-1}"
export LOGDIR="${LOGDIR:-${SCRATCH}/openfold-ref/results/${SLURM_JOB_ID}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/openfold-ref/profile/opt-dataloader-pq-${SLURM_JOB_ID}}"
export DATA_DIR="${DATA_DIR:-${SCRATCH}/openfold/pdb_data}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SCRATCH}/openfold/mlperf_hpc_openfold_resumable_checkpoint_v2.pt}"

# Other settings
export MASTER_ADDR=$(hostname)
export FI_MR_CACHE_MONITOR=userfaultfd
export OMP_NUM_THREADS=1
export TRITON_CACHE_DIR="/dev/shm/triton/job${SLURM_JOBID}/rank${SLURM_PROCID}"

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
    srun -u \
        bash scripts/run_training.sh --distributed   ${args}
done
~            
