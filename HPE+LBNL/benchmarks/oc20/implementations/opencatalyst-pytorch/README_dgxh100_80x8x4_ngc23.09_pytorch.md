## Steps to launch training

### NVIDIA DGX H100 (multi node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX H100
multi node submission are in the `config_DGXH100_80x8x4.sh`script.

Steps required to launch multi node training on NVIDIA DGX H100 80G

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry>/mlperf-nvidia:oc20_pytorch .
docker push <docker/registry>/mlperf-nvidia:oc20_pytorch
```

2. Launch the training

Create a file named `config_data.sh` under the configs directory. Inside that file, specify the following environment variable `export DATADIR=<path/to/dataset>`, then select config and launch the job:

```
source configs/config_DGXH100_80x8x4.sh
CONT="<docker/registry>/mlperf-nvidia:oc20_pytorch" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

#### Alternative launch with nvidia-docker

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
source config_DGXH100_80x8x4.sh
CONT=<docker/registry>mlperf-nvidia:oc20_pytorch DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_and_time.sh
```
