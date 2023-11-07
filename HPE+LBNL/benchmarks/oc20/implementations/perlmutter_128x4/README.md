# Perlmutter 128x4x4 PyTorch OpenCatalyst

OpenCatalyst closed-devision training submission on 128 nodes x 4 GPUs.

In the `../opencatalyst-pytorch` directory, run with
```
source configs/config_pm_128x4x4.sh
sbatch -N $DGXNNODES -t $WALLTIME run_pm.sub
```
