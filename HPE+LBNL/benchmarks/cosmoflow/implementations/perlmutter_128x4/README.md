# Perlmutter 128x4x1 PyTorch CosmoFlow

CosmoFlow closed-devision training submission on 128 nodes x 4 GPUs.

In the `../cosmoflow-pytorch` directory, run with
```
source configs/config_pm_128x4x1.sh
sbatch -N $DGXNNODES -t $WALLTIME run_pm.sub
```
