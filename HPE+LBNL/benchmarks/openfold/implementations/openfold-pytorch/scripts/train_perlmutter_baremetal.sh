#!/bin/bash
#SBATCH -J mlperf-openfold-optimized
#SBATCH -A dasrepo
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=/pscratch/sd/s/schheda/openfold-ref/nvidia/results/openfold_sterr_baremetal_pq_myopt_cu12.log

# Vars with defaults

module unload cray-dsmml cray-libsci perftools-base xalt
module unload craype-accel-nvidia80

module swap cudatoolkit/11.7 cudatoolkit/12.0
module unload Nsight-Compute Nsight-Systems 


module load cudnn/8.9.3_cuda12 nccl/2.18.3-cu12
#module load cray-hdf5-parallel/1.12.2.3
#module swap gcc/11.2.0 gcc/12.2.0 

#module use ~/modulefiles
#module load tcmalloc

export PREFIX=${SCRATCH}/sw-env
export LD_LIBRARY_PATH=${PREFIX}/opt23/lib:${LD_LIBRARY_PATH}
export PATH=${PREFIX}/opt23/bin:${PATH}:/sbin
source ${PREFIX}/env23/bin/activate

NEXP="${NEXP:-1}"
export LOGDIR="${LOGDIR:-${SCRATCH}/openfold-ref/results/myrepo-${SLURM_JOB_ID}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/openfold-ref/profile/myrepo-opt-dataloader-pq-${SLURM_JOB_ID}}"
export DATA_DIR="${DATA_DIR:-/pscratch/sd/s/sfarrell/openfold-ref/data/pdb_data}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SCRATCH}/openfold-ref/mlperf_hpc_openfold_resumable_checkpoint_v2.pt}"

# Other settings
export MASTER_ADDR=$(hostname)
export FI_MR_CACHE_MONITOR=userfaultfd
export OMP_NUM_THREADS=1

export TRITON_CACHE_DIR="/dev/shm/triton/job${SLURM_JOBID}/rank${SLURM_PROCID}"
#export TRITON_CACHE_DIR="/tmp/triton/job${SLURM_JOBID}/rank${SLURM_PROCID}"
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
        bash scripts/run_training.sh --distributed  ${args}

done
~            
