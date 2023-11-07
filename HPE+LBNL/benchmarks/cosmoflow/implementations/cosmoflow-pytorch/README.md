# Cosmoflow benchmark implementation

This repository contains PyTorch implementation of the Cosmoflow: Using Deep Learning to learn the universe at scale benchmark for MLPerf purpose.
Benchmark is based on publicly available reference code: [https://github.com/sparticlesteve/cosmoflow-benchmark](https://github.com/sparticlesteve/cosmoflow-benchmark) 
and the paper: [https://arxiv.org/abs/1808.04728](https://arxiv.org/abs/1808.04728)


## Dataset

The dataset we use for this benchmark comes from simulations run by the ExaLearn group and hosted at NERSC. The following web portal describes the technical content of the dataset and provides links to the raw data.

[https://portal.nersc.gov/project/m3363/](https://portal.nersc.gov/project/m3363/)

For this benchmark we currently use a preprocessed version of the dataset which generates crops of size (128, 128, 128, 4) and stores in TFRecord format. This preprocessing is done using the prepare.py script included in this package. We describe here how to get access to this processed dataset, but please refer to the ExaLearn web portal for additional technical details.

Globus is the current recommended way to transfer the dataset locally. There is a globus endpoint at:

[https://app.globus.org/file-manager?origin_id=d0b1b73a-efd3-11e9-993f-0a8c187e8c12&origin_path=%2F](https://app.globus.org/file-manager?origin_id=d0b1b73a-efd3-11e9-993f-0a8c187e8c12&origin_path=%2F)

The contents are also available via HTTPS at:

[https://portal.nersc.gov/project/dasrepo/cosmoflow-benchmark/](https://portal.nersc.gov/project/dasrepo/cosmoflow-benchmark/)


### Preprocessing

Dataset by default is available in form of GZIP compressed TFRecrods. This implementation uses numpy .npy for storing samples and regression targets.
To convert `*.tfrecrod` to correct `*.npy` format use `tool/convert_tfrecord_to_numpy.py` script.

Exemplary usage:
```
python -m tool.convert_tfrecord_to_numpy -i <data_dir>/train -o <destination_dir>/train -c gzip
```
Additionally, script can benefit from multiple MPI processes, to parallelize the processing.
List of file and labels for validation and training diractory has to be created separately. To do this, execute following bash command:
```
ls -1 <path_to_directory>/ | grep "_data.npy" | sort > <path_to_directory>/files_data.lst
ls -1 <path_to_directory>/ | grep "_label.npy" | sort > <path_to_directory>/files_label.lst
```

For preprocessing of the dataset with SLURM, `init_datasets.sub` job was created. Specify environment variables `DATA_SRC_DIR` and `DATA_DST_DIR` and launch preprocessing with `sbatch -N<preprocessing nodes> init_datasets.sub`.

### Compressed Dataset

Changes have been made to reduce staging time by using a compressed dataset.  Use the command in the pytorch/tools directory on the validation and training datasets to create new compressed datasets

[root]/cosmoflow/pytorch/tools/convert_tfrecord_to_gzip.py

## Before you run
A Dockerfile is provided under the root directory. The following instructions assume you have built the docker container, that you are using slurm with pyxis/enroot support. If your system
uses a different technology, please modify the commands accordingly. Furthermore, the dataset should have been converted to numpy data format as described in the paragraph above.

## How to run the benchmark
The benchmark parameters are steered by environment variables. Please see `run.sub` and `run_and_time.sh`  for more information.
In order to submit the benchmark, you need to source a configuration file you want to run. Those are inside the `configs` directory.
You also need to name your container image. Assuming you named the image `mlperf-cosmoflow:v1.0`, then you can submit a job as follows:

```
export CONT=mlperf-cosmoflow:v1.0
export LOGDIR=<directory for output>
export DATADIR=<root directory for the dataset>
source configs/config_DGXA100_16x8x1.sh
sbatch <specify system-dependent additional args here> -N ${DGXNNODES} -t ${WALLTIME} run.sub
```

The parameters `DGXNNODES` and `WALLTIME` need to be set according to the system being run on.
Please not that in order to pass a locally built image to enroot, you need to export it as a `sqsh` file using `enroot import/create` (see the enroot documentation for instructions). 
We recommend using a registry as enroot can pull the image directly from there.

### Strong scaling benchmark run
To reproduce results for strong scaling 128 nodes, use the following sequence of commands:
```
export CONT=mlperf-cosmoflow:v1.0
export LOGDIR=results/128x8x1/
export DATADIR=<root directory for the dataset>
export DGXNODES=128
source configs/config_DGXA100_128x8x1_other.sh
export CONFIG_FILE=submission_dgxa100_128x8x1_other
sbatch <specify system-dependent additional args here> -N ${DGXNNODES} -t ${WALLTIME} run.sub
```

To reproduce results for strong scaling 16 nodes, use the following sequence of commands:
```
export CONT=mlperf-cosmoflow:v1.0
export LOGDIR=results/16x8x1/
export DATADIR=<root directory for the dataset>
export DGXNODES=16
source configs/config_DGXA100_16x8x1.sh
export CONFIG_FILE=submission_dgxa100_16x8x1
sbatch <specify system-dependent additional args here> -N ${DGXNNODES} -t ${WALLTIME} run.sub
```


### Weak scaling benchmark run
To reproduce results for weak scaling 4 nodes per instance, 512 nodes in total, use the following sequence of commands:
```
export CONT=mlperf-cosmoflow:v1.0
export LOGDIR=results/128x4x8x1_weak/
export DATADIR=<root directory for the dataset>
export DGXNODES=512   # 4 nodes x 128 instances
export CONFIG_FILE=submission_dgxa100_4x8x1
sbatch <specify system-dependent additional args here> -N ${DGXNNODES} -t ${WALLTIME} run.sub +instances=128
```

You can modify total number of parallel instances using `NUM_INSTANCES` environmental variable.

## Hyperparameters

The table below contains the modifiable hyperparameters. Unless otherwise stated, parameters not listed in the table below are fixed and changing those could lead to an invalid submission.

| Parameter Name          | Default    | Constraints                     | Description |
| ----------------------- | ---------- | ------------------------------- | ----------- |
| `--seed`                | None       | >= 0                            | Random number generator seed. Do not use the same seed for multiple submission except for debugging purpose. |
| `--initial-lr`          | 0.001      | >= 0                            | Learning rate from which to start training. This is before the warmup happens and learning rate increases from that point during warmup epochs |
| `--base-lr`             | 0.008      | >= `initial-lr`                 | Base learning rate to use after the warmup is completed. |
| `--weight-decay`        | 0.0        | >= 0 if `dropout` = 0           | Weight decay to use instead of dropout for dense layers of the network |
| `--dropout`             | 0.5        | 1 > x > 0 if `weight-decay` = 0 | Dropout rate used for dense layers instead of weight decay |
| `--warmup-epochs`       | 4          | >= 0                            | Number of epochs for LR warmup during which lr is increasing from `initial-lr` to `base-lr` |
| `--lr-scheduler-epochs` | 32 64      | >= 0, count = 2                 | Epochs on which learning rate should be decreased by the specified factor |
| `--lr-scheduler-decays` | 0.25 0.125 | 1 > x > 0, count = 2            | Factors that reduces LR on specified epochs | 
| `--shuffle`             | False      | boolean                         | Specify if you want to shuffle dataset after each epoch |


**Please note that besides the benchmark specific rules above, standard MLPerf HPC logging rules apply.**

## References
1. A. Mathuriya, D. Bard, P. Mendygral, L. Meadows, J. Arnemann, L. Shao, S. He, T. Karna, D. Moise, S. J. Pennycook, K. Maschoff, J. Sewall, N. Kumar, S. Ho, M. Ringenburg, Prabhat, and V. Lee:
CosmoFlow: Using Deep Learning to Learn the Universe at Scale, SuperComputing 2018
