#!/bin/bash
#SBATCH -J oc-preproc
#SBATCH -q regular
#SBATCH -t 4:00:00
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -c 256
#SBATCH --image schheda/opencatalyst-opt:23.07
#SBATCH -o slurm-%x-%j.out

srun shifter python scripts/make_lmdb_sizes.py \
    --data-path /pscratch/sd/s/sfarrell/optimized-hpc/opencatalyst/data/oc20_data/s2ef/2M/train \
    --num-workers 128

srun shifter python scripts/make_lmdb_sizes.py \
    --data-path /pscratch/sd/s/sfarrell/optimized-hpc/opencatalyst/data/oc20_data/s2ef/all/val_id \
    --num-workers 128
