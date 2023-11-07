#!/bin/bash
#
# Copyright 2023 NVIDIA CORPORATION
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
#
# Usage: sbatch scripts/multi_node_training.sub

#SBATCH --job-name mlperf-hpc:openfold-reference
#SBATCH -N 18                   # number of nodes
#SBATCH -n 18			        # number of processes
#SBATCH -t 72:00:00             # wall time
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --gres=gpu:2
#SBATCH --reservation=mlperf

module purge
module add openmpi/4.1.5

# Print current datetime:
echo "START" $(date +"%Y-%m-%d %H:%M:%S")

# Print node list:
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_NODELIST=$SLURM_NODELIST"

# Note: the following srun commands assume that pyxis plugin is installed on a SLURM cluster.
# https://github.com/NVIDIA/pyxis
export CONT=/scratch/nnisbet/mlperf_hpc-openfold_latest.sif

#srun \
#--mpi=none \
#apptainer exec --nv -B /scratch/nnisbet/openfold:/data:rw,$PWD:/training_rundir \
#$CONT \
#nvidia-smi

# Print current datetime again:
echo "READY" $(date +"%Y-%m-%d %H:%M:%S")

# Set number of threads to use for parallel regions:
export OMP_NUM_THREADS=1
export UCX_POSIX_USE_PROC_LINK=n
export NCCL_ASYNC_ERROR_HANDLING=1

# Set MLPerf variables:
export DATESTAMP=$(date +"%y%m%d%H%M%S%N")
export EXP_ID=1

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

for _experiment_index in $(seq 1 10); do
(
	# Clear caches
	srun --ntasks="${SLURM_JOB_NUM_NODES}" bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"

	export SEED=${_seed_override:-$(date +%s)}
	# Run the command:
	srun \
	--mpi=pmi2 \
	apptainer exec --nv -B /etc/hosts:/etc/hosts,/scratch/nnisbet/openfold:/data:rw,$PWD:/training_rundir \
	$CONT \
	bash -c \
	'echo "srun SLURMD_NODENAME=$SLURMD_NODENAME MASTER_ADDR=$MASTER_ADDR"; \
	torchrun \
	--nnodes=$SLURM_JOB_NUM_NODES \
	--nproc_per_node=2 \
	--rdzv_id=$SLURM_JOB_ID \
	--rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_ADDR \
	/training_rundir/train.py \
	--training_dirpath /training_rundir \
	--pdb_mmcif_chains_filepath /data/pdb_mmcif/processed/chains.csv \
	--pdb_mmcif_dicts_dirpath /data/pdb_mmcif/processed/dicts \
	--pdb_obsolete_filepath /data/pdb_mmcif/processed/obsolete.dat \
	--pdb_alignments_dirpath /data/open_protein_set/processed/pdb_alignments \
	--initialize_parameters_from /data/mlperf_hpc_openfold_resumable_checkpoint.pt \
	--seed $SEED \
	--num_train_iters 2000 \
	--val_every_iters 40 \
	--local_batch_size 4 \
	--base_lr 1e-3 \
	--warmup_lr_init 1e-5 \
	--warmup_lr_iters 0 \
	--num_train_dataloader_workers 16 \
	--num_val_dataloader_workers 2 \
	--distributed'
) |& tee "slurm_${DATESTAMP}_${_experiment_index}.out"
done