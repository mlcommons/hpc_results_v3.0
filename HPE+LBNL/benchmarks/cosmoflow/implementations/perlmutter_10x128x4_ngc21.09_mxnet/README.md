# Perlmutter 10x128x4 MXNet CosmoFlow

CosmoFlow weak-scaling closed-devision submission on 1280 nodes x 4 GPUs.
10 models are trained concurrently each using 128 nodes x 4 GPUs.

To run:

```
source configs/config_pm_10x128x4x1_weak.sh
sbatch -N $DGXNNODES -t $WALLTIME run_pm.sub
```
