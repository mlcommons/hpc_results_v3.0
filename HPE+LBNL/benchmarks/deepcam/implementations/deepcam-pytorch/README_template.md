#  JOB_SHAPE PyTorch DeepCAM

DeepCAM SCALING_TYPE-scaling closed-devision submission on NUM_NODES nodes x NUM_GPU GPUs with batch size GLOBAL_BATCH_SIZE.

To run:

```
export CONT=mlperf-deepcam:v2.0
source configs/CONFIG_FILE
sbatch -N $DGXNNODES -t $WALLTIME run.sub
```