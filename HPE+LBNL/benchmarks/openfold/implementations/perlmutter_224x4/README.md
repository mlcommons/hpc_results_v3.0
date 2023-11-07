# Perlmutter 224x4x1x4\_dap PyTorch OpenFold

OpenFold closed-devision training submission on 224 nodes x 4 GPUs.

In the `../openfold-pytorch` directory, run with
```
source configs/config_pm_224x4x1x4_dap.sh
sbatch -N $DGXNNODES -t $WALLTIME run_pm.sub
```
