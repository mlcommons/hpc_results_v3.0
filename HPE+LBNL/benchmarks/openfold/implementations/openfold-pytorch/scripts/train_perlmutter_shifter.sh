#!/bin/bash
#SBATCH -J mlperf-openfold-optimized
#SBATCH -A m4291_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=openfold_shifter_224x4x1_4dap_7.log
#SBATCH --image=schheda/openfold:23.09
# --image=registry.nersc.gov/das/openfold-opt:23.09.01
#SBATCH --module=nccl-2.18,gpu
# Vars with defaults

#export PREFIX=${SCRATCH}/sw-env
#export LD_LIBRARY_PATH=${PREFIX}/opt23/lib:${LD_LIBRARY_PATH}
#export PATH=${PREFIX}/opt23/bin:${PATH}:/sbin
#source ${PREFIX}/env23/bin/activate

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
    srun -u --mpi=pmi2 shifter \
        bash scripts/run_training.sh --distributed  ${args}

done
~            
