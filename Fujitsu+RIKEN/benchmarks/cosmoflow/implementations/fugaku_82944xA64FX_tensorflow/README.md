# 1. Problem

This problem uses TensorFlow and Mesh TensorFlow implementations for the cosmological parameter prediction benchmark.

# 2. Directions
## Steps to download and verify data
Download the data as follows:

Please download the dataset manually following the instructions from the [CosmoFlow TensorFlow Keras benchmark implementation](https://github.com/mlcommons/hpc/tree/main/cosmoflow).

## Steps to reformat the dataset

Steps required to decompress TFRecords and compress them with XZ.

```
./init_datasets.sh
```

## Steps to launch training

### The supercomputer Fugaku
Steps required to launch training on Fugaku:

```
./setup.sh
./run_and_time.sh
```

# Note

## Measurement and Result Description
Data parallel training is used (hybrid data-model parallel training is not used).

Weak scaling is measured using 82,944 nodes (= 512 nodes x 162 model instances).

compliance_checker, system_desc_checker, and package_checker passed (with a minor fix in rcp_checker.py).

### Closed division
- Weak scaling: 147 runs are registered as a result.
- Strong scaling: the first 10 runs of weak scaling are registered as a result.

### Open division
The kernel size of Conv3D layers has been changed to 2 (not allowed as hyperparameter in closed division).
- Weak scaling: 81 runs are registered as a result.
- Strong scaling: the first 10 runs of weak scaling are registered as a result.

## Implementation Description
This Mesh TensorFlow implementation has been updated from the v0.7 implementation.

### For compliance with v1.0
- Support kernel-size=3 for Conv3D layers.
- Add weight decay in SGD optimizer only for kernel weights of dense layers  
  Weight decay on SGD optimizer is equivalent to L2 regularizer if the weight decay is twice the value of the L2 regularizer (e.g., weight decay = 0.02 is equivalent to L2 regularizer = 0.01).
- Add some MLLOGs.  
  The value of `opt_weight_decay` shows half the value of weight decay because the reference implementation shows L2 regularizer value as `opt_weight_decay`.

### For performance improvement
- Eliminate unnecessary halo processing when data parallel only training

### For weak scaling measurement
- Add inter model-instance synchronization to align `run_start` with other instances.  
  Due to system limitations, instances were run in multiple jobs. Each instance in all jobs execute program until just before `run_start`, creates a file on the shared storage to notify that it is ready, and waits for a start signal. When all instances are ready, we create a file on the shared storage to notify all instances of start.

- Add data staging for multi-instances  
  We made a MPI program for multi-instance staging. The program is launched on all nodes in a job, concurrently with the instances. When the instances notify the program of staging start, only the nodes belonging to the first instance read the training data from the shared storage and broadcast it to the nodes belonging to the other instances. When staging is complete, the program notifies the instances of it, and the instances start training.

### Others
- Each instance initializes model weights with random numbers based on a seed that is get by $RANDOM of Bash.
- The implementation supports GZIP compressed TFRecord dataset, but we use uncompressed TFRecord dataset in the measurement.
- The implementation can overlap gradient communication with backward propagation computation, but the feature is disabled in the measurement because it had no desired effect.
