#!/bin/bash
#SBATCH --job-name mlperf-hpc-openfold
#SBATCH -N  32
#SBATCH --ntasks-per-node 3
#SBATCH --cpus-per-task 32
#SBATCH -p gpu-a100
#SBATCH --output=slurm-%j.txt
#SBATCH --time=6:00:00
#SBATCH --reservation=mlcommons


# Print current datetime:
echo "started at `date`"
echo "START" $(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

# Initialization
ml reset
ml gcc/11.2.0 cuda/12.0 nccl cudnn
module list
source scripts/activate_local_openfold_venv.sh /scratch/05231/aruhela/mlcommons/openfold-venv/
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=".10"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"

# Optional Information
which python 
which python3 
ldd `which python`
python -m torch.utils.collect_env
python -c 'import torch ; print(torch.__version__)'
python -c "import torch; print(torch.cuda.is_available())"
pip list
conda list
env | grep SLURM
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
# Print node list:
echo "SLURM_NODELIST=$SLURM_NODELIST"

echo "Clean any stray Python Processes from previous jobs, if any"
mpiexec -np $SLURM_NNODES -ppn 1 pkill python

cd /scratch/05231/aruhela/mlcommons/hpc/openfold
mydir="output-$SLURM_JOB_ID"
outdir="/scratch/05231/aruhela/mlcommons/hpc/openfold/$mydir"
datadir=/work/05231/aruhela/ls6/openfold

# Print current datetime again:
echo "READY" $(date +"%Y-%m-%d %H:%M:%S")

# Set number of threads to use for parallel regions:
#export OMP_NUM_THREADS=1
unset OMP_NUM_THREADS

# Set MLPerf variables:
export DATESTAMP=$(date +"%y%m%d%H%M%S%N")
export EXP_ID=1
RANDOM=$(date +%s)
echo "RANDOM=$RANDOM"

# Setting nodes, ranks, and hostfile
ppn=3
nodes=$SLURM_NNODES
ranks=$((ppn*nodes))
export IBRUN_TASKS_PER_NODE=$ppn
echo "PPN   = $ppn"
echo "IBRUN_TASKS_PER_NODE=$IBRUN_TASKS_PER_NODE"
echo "Ranks = $ranks"
echo "Nodes = $SLURM_NNODES"
# Generate HostFile
gen_hostfile.sh $ppn
myhostfile="hosts.$SLURM_JOBID"
cat hosts

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=29500
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
export MY_MPIRUN_OPTIONS="-env MASTER_ADDR=$MASTER_ADDR -env MASTER_PORT=$MASTER_PORT "
export TACC_IBRUN_DEBUG=1
export I_MPI_DEBUG=4

# Run the Benchmark
set -x
/usr/bin/time -f "real \t%e (seconds)" \
mpiexec -np $ranks -f $myhostfile -ppn $ppn \
-genv I_MPI_PIN_DOMAIN [0000000000000000FFFFFFFFFFFFFFFF,FFFFFFFFFFFFFFFF0000000000000000,FFFFFFFFFFFFFFFF0000000000000000] \
-print-rank-map -env I_MPI_DEBUG=4 \
/scratch/05231/aruhela/mlcommons/openfold-venv/conda/envs/openfold-venv/bin/python \
train.py \
--training_dirpath $outdir \
--pdb_mmcif_chains_filepath $datadir/pdb_data/pdb_mmcif/processed/chains.csv \
--pdb_mmcif_dicts_dirpath $datadir/pdb_data/pdb_mmcif/processed/dicts \
--pdb_obsolete_filepath $datadir/pdb_data/pdb_mmcif/processed/obsolete.dat \
--pdb_alignments_dirpath $datadir/pdb_data/open_protein_set/processed/pdb_alignments \
--initialize_parameters_from $datadir/mlperf_hpc_openfold_resumable_checkpoint_b518be46.pt \
--seed $RANDOM \
--num_train_iters 2000 \
--val_every_iters 40 \
--local_batch_size 1 \
--base_lr 1e-3 \
--warmup_lr_init 1e-5 \
--warmup_lr_iters 0 \
--num_train_dataloader_workers 12 \
--num_val_dataloader_workers 3 \
--distributed \
--gradient_accumulation_iters 2 --log_every_iters 2
set +x

echo -e "`date` : ----- FINISHED $me in $SECONDS Seconds-------"

