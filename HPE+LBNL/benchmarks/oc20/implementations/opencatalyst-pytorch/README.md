# Open Catalyst Project models

## MLPerf HPC benchmark instructions

This repository defines the reference implementation for the MLPerf HPC
OpenCatalyst DimeNet++ benchmark. Instructions are below for general
usage of the repository. The reference model configuration can be found in
[configs/mlperf\_hpc.yml](configs/mlperf_hpc.yml).

To download the data, you can use the provided download script as described
below. The commands to prepare the data in the same way as the reference are

```bash
# Download the 2M training dataset with edge info
python scripts/download_data.py --task s2ef --split 2M --get-edges --num-workers 64 --ref-energy
# Download the val-id evaluation dataset
python scripts/download_data.py --task s2ef --split val_id --get-edges --num-workers 64 --ref-energy
```

For convenience, we are distributing the preprocessed dataset as a tarball
- via [globus endpoint](https://app.globus.org/file-manager?origin_id=7008326a-def1-11eb-9b4d-47c0f9282fb8&origin_path=%2F)
- via [NERSC web portal](https://portal.nersc.gov/project/dasrepo/mlperf-oc20/)

## Introduction

ocp-models is the modeling codebase for the [Open Catalyst Project](https://opencatalystproject.org/).

It provides implementation of [DimeNet++](https://arxiv.org/abs/2011.14115), ML algorithms for catalysis that
take arbitrary chemical structures as input to predict energy / forces / positions:

## Installation

The repository contains Dockerfile which handles all required dependencies. Here are the steps to prepare the environment:

1. Clone repository
2. Build docker image, `./docker/build.sh`
3. Run docker container, `./docker/run.sh`. You have to export DATA envvar with the path to the dataset on your disk.
4. Start training, for example, `python main.py --batch_size 2 --eval_batch_size 2`

first create a file named `config_data.sh` under the configs directory. Inside that file, specify the following environment variable `export DATADIR=<path/to/dataset>`.


To run on cluster with SLURM scheduler, first create a file named `config_data.sh` under the configs directory. Inside that file, specify the following environment variable `export DATADIR=<path/to/dataset>`. Then, you have to source the hparam in the config direcotry, for example: `source configs/config_DGXA100_8x8x16.sh`. 

Additionally, you have to export the following environment variables:
- CONT: container name to be pulled from registry,
- DATADIR: path to the data directory,
- LOGDIR: path to the log directory,
- DGXNODES: number of nodes,
- WALLTIME: slurm time limit.

For example:

```
source configs/config_DGXA100_8x8x16.sh
CONT=<container/name> DATADIR=<path/to/data> LOGDIR=<path/to/log> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

## Download data

Dataset download links for all tasks can be found at [DATASET.md](https://github.com/Open-Catalyst-Project/ocp/blob/master/DATASET.md).

IS2* datasets are stored as LMDB files and are ready to be used upon download.
S2EF train+val datasets require an additional preprocessing step.

For convenience, a self-contained script can be found [here](https://github.com/Open-Catalyst-Project/ocp/blob/master/scripts/download_data.py) to download, preprocess, and organize the data directories to be readily usable by the existing [configs](https://github.com/Open-Catalyst-Project/ocp/tree/master/configs).

For IS2*, run the script as:

```bash
python scripts/download_data.py --task is2re
```

For S2EF train/val, run the script as:

```bash
python scripts/download_data.py --task s2ef --split SPLIT_SIZE --get-edges --num-workers WORKERS --ref-energy
```

- `--split`: split size to download: `"200k", "2M", "20M", "all", "val_id", "val_ood_ads", "val_ood_cat", or "val_ood_both"`.
- `--get-edges`: includes edge information in LMDBs (~10x storage requirement, ~3-5x slowdown), otherwise, compute edges on the fly (larger GPU memory requirement).
- `--num-workers`: number of workers to parallelize preprocessing across.
- `--ref-energy`: uses referenced energies instead of raw energies.

For S2EF test, run the script as:
```bash
python scripts/download_data.py --task s2ef --split test
```

## Train and evaluate models

A detailed description of how to train and evaluate models, run ML-based
relaxations, and generate EvalAI submission files can be found
[here](https://github.com/Open-Catalyst-Project/ocp/blob/master/TRAIN.md).

Our evaluation server is [hosted on EvalAI](https://eval.ai/web/challenges/challenge-page/712/overview).
Numbers (in papers, etc.) should be reported from the evaluation server.

## Pretrained models

Pretrained model weights accompanying [our paper](https://arxiv.org/abs/2010.09990) are available [here](https://github.com/Open-Catalyst-Project/ocp/blob/master/MODELS.md).

### Tutorials

Interactive tutorial notebooks can be found [here](https://github.com/Open-Catalyst-Project/ocp/tree/master/docs/source/tutorials) to help familirize oneself with various components of the repo:

- [Data visualization](https://github.com/Open-Catalyst-Project/ocp/blob/tutorials/docs/source/tutorials/data_visualization.ipynb) - understanding the raw data and its contents.
- [Data preprocessing](https://github.com/Open-Catalyst-Project/ocp/blob/tutorials/docs/source/tutorials/data_preprocessing.ipynb) - preprocessing raw ASE atoms objects to OCP graph Data objects.
- [LMDB dataset creation](https://github.com/Open-Catalyst-Project/ocp/blob/tutorials/docs/source/tutorials/lmdb_dataset_creation.ipynb) - creating your own OCP-compatible LMDB datasets from ASE-compatible Atoms objects.
- [S2EF training example](https://github.com/Open-Catalyst-Project/ocp/blob/tutorials/docs/source/tutorials/train_s2ef_example.ipynb) - training a SchNet S2EF model, loading a trained model, and making predictions.

## Discussion

For all non-codebase related questions and to keep up-to-date with the latest OCP announcements, please join the [discussion board](https://discuss.opencatalystproject.org/). All codebase related questions and issues should be posted directly on our [issues page](https://github.com/Open-Catalyst-Project/ocp/issues).

## Acknowledgements

- This codebase was initially forked from [CGCNN](https://github.com/txie-93/cgcnn)
by [Tian Xie](http://txie.me), but has undergone significant changes since.
- A lot of engineering ideas have been borrowed from [github.com/facebookresearch/mmf](https://github.com/facebookresearch/mmf).
- The DimeNet++ implementation is based on the [author's Tensorflow implementation](https://github.com/klicperajo/dimenet) and the [DimeNet implementation in Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py).

## Citation

If you use this codebase in your work, consider citing:

```bibtex
@article{ocp_dataset,
    author = {Chanussot*, Lowik and Das*, Abhishek and Goyal*, Siddharth and Lavril*, Thibaut and Shuaibi*, Muhammed and Riviere, Morgane and Tran, Kevin and Heras-Domingo, Javier and Ho, Caleb and Hu, Weihua and Palizhati, Aini and Sriram, Anuroop and Wood, Brandon and Yoon, Junwoong and Parikh, Devi and Zitnick, C. Lawrence and Ulissi, Zachary},
    title = {Open Catalyst 2020 (OC20) Dataset and Community Challenges},
    journal = {ACS Catalysis},
    year = {2021},
    doi = {10.1021/acscatal.0c04525},
}
```

## License

[MIT](https://github.com/Open-Catalyst-Project/ocp/blob/master/LICENSE.md)
