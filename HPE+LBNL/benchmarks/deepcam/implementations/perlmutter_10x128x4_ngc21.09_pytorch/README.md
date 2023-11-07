# Perlmutter 10x128x4x2 PyTorch DeepCAM

DeepCAM weak-scaling closed-devision submission with 10 models trained on
128 nodes x 4 GPUs. 1280 nodes total.

To run:

```
source configs/config_pm_10x128x4x2_weak.sh
sbatch -N $DGXNNODES -t $WALLTIME run_pm.sub
```
