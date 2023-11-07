# Perlmutter 512x4x1 PyTorch DeepCAM

DeepCAM closed-devision training submission on 512 nodes x 4 GPUs.

In the `../deepcam-pytorch` directory, run with
```
source configs/config_pm_128x4x1.sh
sbatch -N $DGXNNODES -t $WALLTIME run_pm.sub
```
