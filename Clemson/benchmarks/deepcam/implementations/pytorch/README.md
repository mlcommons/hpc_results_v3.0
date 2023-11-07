# Deep Learning Climate Segmentation Benchmark

PyTorch implementation for the climate segmentation benchmark, based on the
Exascale Deep Learning for Climate Analytics codebase here:
https://github.com/azrael417/ClimDeepLearn, and the paper:
https://arxiv.org/abs/1810.01993

## Dataset

The dataset for this benchmark comes from CAM5 [1] simulations and is hosted at
NERSC. The samples are stored in HDF5 files with input images of shape
(768, 1152, 16) and pixel-level labels of shape (768, 1152). The labels have
three target classes (background, atmospheric river, tropical cycline) and were
produced with TECA [2].

The current recommended way to get the data is to use GLOBUS and the following
globus endpoint:

https://app.globus.org/file-manager?origin_id=0b226e2c-4de0-11ea-971a-021304b0cca7&origin_path=%2F

The dataset folder contains a README with some technical description of the
dataset and an All-Hist folder containing all of the data files.

### Preprocessing
The dataset is split into train/validation/test and ships with the `stats.h5` file containing summary statistics.
In order to run the benchmark with the various DALI readers, the dataset has to be converted into numpy file format.
For this purpose, the script `src/utils/convert_hdf52npy.py` is provided (`/opt/utils/convert_hdf52npy.py` inside the container). This script reads the original HDF5 files and generates numpy files for data and labels.
The script leverages MPI to performa distributed conversion of the files and needs to be run for each file directory separately, e.g. for validation and training. Example:

```
DATA_IN=<hdf5-data-path>
DATA_OUT=<numpy-data-path>
NUM_TASKS=1

cp ${DATA_IN}/stats.h5 ${DATA_OUT}/
mpirun -np ${NUM_TASKS} python src/utils/convert_hdf52npy.py --input_directory=${DATA_IN}/train      --output_directory=${DATA_OUT}/train
mpirun -np ${NUM_TASKS} python src/utils/convert_hdf52npy.py --input_directory=${DATA_IN}/validation --output_directory=${DATA_OUT}/validation
```

For docker users with slurm job schedulers and pyxis/enroot support we have added the script `init_datasets.sub`, which performs this conversion. 

## Before you run

A Dockerfile is provided under the `docker` subdirectory. The following instructions assume you have built the docker container, that you are using slurm with pyxis/enroot support. If your system
uses a different technology, please modify the commands accordingly. Furthermore, the dataset should have been converted to numpy data format as described in the paragraph above. Finally create a file named `config_data.sh` under the configs directory. Inside that file, specify the following environment variable:

```
#!/bin/bash

export DATADIR=<root path to where the converted dataset resides>
```
Make this file executable, i.e. `chmod +x configs/config_data.sh`. If you are using docker, the path specified via the `DATADIR` environment variable will be mounted (read-only) into the container under `/data` from where the benchmark will pick it up.

## How to run the benchmark

The benchmark parameters are steered by environment variables. Please see `run.sub` (`/workspace/run.sub` inside the container), `run_and_time.sh` (`/opt/deepCam/run_and_time.sh` inside the container) for more information.
In order to submit the benchmark, you need to source a configuration file you want to run. Those are under `configs` (or under `/workspace/configs` inside the container).
You also need to name your container image. Assuming you named the image `mlperf-deepcam:v1.0`, then you can submit a job as follows:

```
export CONT=mlperf-deepcam:v1.0
source configs/config_DGXA100_16x8x1.sh
sbatch <specify system-dependent additional args here> -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
The parameters `DGXNNODES` and `WALLTIME` are set by the configuration files.

Please not that in order to pass a locally built image to enroot, you need to export it as a `sqsh` file using `enroot import/create` (see the [enroot documentation](https://github.com/NVIDIA/enroot/blob/master/doc/usage.md) for instructions). We recommend using a registry as enroot can pull the image directly from there.


## Hyperparameters

The table below contains the modifiable hyperparameters. Unless otherwise stated, parameters not
listed in the table below are fixed and changing those could lead to an invalid submission.

|Parameter Name |Default | Constraints | Description|
--- | --- | --- | ---
`--optimizer` | `"Adam"` | Optimizer of Adam or LAMB* type. This benchmark implements `"Adam"` and `"AdamW"` from PyTorch as well as `"FusedLAMB"` from NVIDIA APEX. Algorithmic equivalent implementations to those listed before are allowed. | The optimizer to choose
`--start_lr` | 1e-3 | >= 0. | Start learning rate (or base learning rate if warmup is used)
`--optimizer_betas` | `[0.9, 0.999]` | N/A | Momentum terms for Adam-type optimizers
`--weight_decay` | 1e-6 | >= 0. | L2 weight regularization term
`--lr_warmup_steps` | 0 | >= 0 | Number of steps for learning rate warmup
`--lr_warmup_factor` | 1. | >= 1. | When warmup is used, the target learning_rate will be lr_warmup_factor * start_lr
`--lr_schedule` | - | `type="multistep",milestones="<milestone_list>",decay_rate="<value>"` or `type="cosine_annealing",t_max="<value>",eta_min="<value>"` | Specifies the learning rate schedule. Multistep decays the current learning rate by `decay_rate` at every milestone in the list. Note that the milestones are in unit of steps, not epochs. Number and value of milestones and the `decay_rate` can be chosen arbitrarily. For a milestone list, please specify it as whitespace separated values, for example `milestones="5000 10000"`. For cosine annealing, the minimal lr is given by the value of `eta_min` and the period length in number of steps by `T_max`
`--batchnorm_group_size` | 1 | >= 1 | Determines how many ranks participate in the batchnorm. Specifying a value > 1 will replace nn.BatchNorm2d with nn.SyncBatchNorm everywhere in the model. Currently, nn.SyncBatchNorm only supports node-local batch normalization, but using an Implementation of that same functionality which span arbitrary number of workers is allowed
`--gradient_accumulation_frequency` | 1 | >= 1 | Specifies the number of gradient accumulation steps before a weight update is performed
`--seed` | 333 | > 0 | Random number generator seed. Multiple submissions which employ the same seed are discouraged. Please specify a seed depending on system clock or similar.

*LAMB optimizer has additional hyperparameters such as the global grad clipping norm value. For the purpose of this benchmark, consider all those parameters which are LAMB specific and fixed. The defaults are specified in the [NVIDIA APEX documentation for FusedLAMB](https://nvidia.github.io/apex/_modules/apex/optimizers/fused_lamb.html).

Note that the command line arguments do not directly correspond to logging entries. For compliance checking of oiutput logs, use the table below:

|Key| Constraints | Required |
--- | --- | ---
`seed` | `x > 0` | True
`global_batch_size` | `x > 0` | `True`
`num_workers` | `x > 0` | `True`
`batchnorm_group_size` | `x > 1` | `False`
`gradient_accumulation_frequency` | `x >= 1` | `True`
`opt_name` | `x in ["Adam", "AdamW", "LAMB", "MixedPrecisionLAMB", "DistributedLAMB"]` | `True`
`opt_lr` | `x >= 0.` | `True`
`opt_betas` | unconstrained | `True`
`opt_eps` | `x == 1e-6` | `True`
`opt_weight_decay` | `x >= 0.` | `True`
`opt_bias_correction` | `x == True` | `True if optimizer_name == "LAMB" else False`
`opt_grad_averaging` | `x == True` | `True if optimizer_name == "LAMB" else False`
`opt_max_grad_norm` | `x == 1.0` | `True if optimizer_name == "LAMB" else False`
`scheduler_type` | `x in ["multistep", "cosine_annealing"]` | `True`
`scheduler_milestones` | unconstrained | `True if scheduler_type == "multistep" else False`
`scheduler_decay_rate` | `x >= 1.` | `True if scheduler_type == "multistep" else False`
`scheduler_t_max` | `x >= 0` | `True if scheduler_type == "cosine_annealing" else False`
`scheduler_eta_min` | `x >= 0.` | `True if scheduler_type == "cosine_annealing" else False`
`scheduler_lr_warmup_steps` | `x >= 0` | `False`
`scheduler_lr_warmup_factor` | `x >= 1.` | `True if scheduler_lr_warmup_steps > 0 else False`

The first column lists the keys as they would appear in the logfile. The second column lists the parameters constraints as an equation for parameter variable x. Those can be used to generate lambda expressions in Python. The third one if the corresponding entry has to be in the log file or not. Since there are multiple optimizers and learning rate schedules to choose from, not all parameters need to be logged for a given run. This is expressed by conditional expressions in that column.
**Please note that besides the benchmark specific rules above, standard MLPerf HPC logging rules apply.**

### Using Docker

The implementation comes with a Dockerfile optimized for NVIDIA workstations but usable on 
other NVIDIA multi-gpu systems. Use the Dockerfile 
`docker/Dockerfile.train` to build the container and the script `src/deepCam/run_scripts/run_training.sh`
for training. The data_dir variable should point to the full path of the `All-Hist` directory containing the downloaded dataset.

## References

1. Wehner, M. F., Reed, K. A., Li, F., Bacmeister, J., Chen, C.-T., Paciorek, C., Gleckler, P. J., Sperber, K. R., Collins, W. D., Gettelman, A., et al.: The effect of horizontal resolution on simulation quality in the Community Atmospheric Model, CAM5. 1, Journal of Advances in Modeling Earth Systems, 6, 980-997, 2014.
2. Prabhat, Byna, S., Vishwanath, V., Dart, E., Wehner, M., Collins, W. D., et al.: TECA: Petascale pattern recognition for climate science, in: International Conference on Computer Analysis of Images and Patterns, pp. 426-436, Springer, 2015b.
