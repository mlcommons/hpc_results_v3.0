import argparse
import enum
import logging
import time
import math
import pathlib
import random
import os
from types import resolve_bases

from typing import Callable, Union, List, Tuple, Literal

import numpy as np
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

import utils


from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as dali_fn
import nvidia.dali.math as dali_math
import nvidia.dali.types as dali_types

import nvidia.dali.plugin.mxnet as dali_mxnet


@utils.ArgumentParser.register_extension("DALI pipeline")
def add_dali_argument_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--dali-num-threads", type=int, default=1,
                        help="Number of thread per GPU for DALI preprocessing")
    parser.add_argument("--dali-use-mmap", action="store_true",
                        default=False, help="Use mmap instead of standard file operations")


@utils.ArgumentParser.register_extension("Generic data argument")
def add_data_argument_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--data-root-dir", type=pathlib.Path,
                        help="Directory where dataset is located")
    parser.add_argument("--training-samples", type=int, default=-1,
                        help="Number of training samples per training epoch")
    parser.add_argument("--validation-samples", type=int, default=-1,
                        help="Number of validation samples per validation")
    parser.add_argument("--training-batch-size", type=int,
                        default=1, help="Batch size per GPU used for training")
    parser.add_argument("--validation-batch-size", type=int,
                        default=1, help="Batch size per GPU used for validation")
    parser.add_argument("--apply-log-transform", action="store_true",
                        default=False, help="Apply log(x+1) transform for input data")
    parser.add_argument("--data-layout", choices=["NDHWC", "NCDHW"],
                        default="NDHWC", help="Data layout for training and dataloading")
    parser.add_argument("--data-shard-multiplier", type=int, default=1, help="Manual sharding with oversample span")
    parser.add_argument("--shard-type", type=str, choices=["local", "global"], default="global",
                        help="Sharding type for training data, use global sharding or per node sharding")
    parser.add_argument("--prestage", action="store_true", default=False, help="Prestage training data into local ramdisk")
    parser.add_argument("--shuffle", action="store_true", default=False, help="Shuffle after each epoch")
    parser.add_argument("--preshuffle", action="store_true", default=False, help="Shuffle entire dataset before assign samples to each rank/node")


class DatasetDescriptor(object):
    class ShardingMethod(enum.Enum):
        SEQUENTIAL = 0,
        INTERLEAVE = 1

    
    def __init__(self, 
                 dist_desc: utils.DistributedEnvDesc,
                 data_file_list: List[str],
                 label_file_list: List[str]):
        assert len(data_file_list) == len(label_file_list), \
            "Number of data file must be same as number of label files"

        self.dist_desc = dist_desc
        self.data_file_list = data_file_list
        self.label_file_list = label_file_list
        self.samples = len(data_file_list)

    def get_per_node_list(self, node: int = -1) -> Tuple[List[str], List[str]]:
        if node == -1:
            node = self.dist_desc.rank // self.dist_desc.local_size


    @staticmethod
    def _divide_to_chunks(data: List[str], shard_id: int = 0, num_shards: int = 1, 
                          method: ShardingMethod = ShardingMethod.SEQUENTIAL) -> List[str]:
        if method == DatasetDescriptor.ShardingMethod.SEQUENTIAL:
            per_shard_items = len(data) // num_shards
            return data[shard_id * per_shard_items : (shard_id+1) * per_shard_items]
        elif method == DatasetDescriptor.ShardingMethod.INTERLEAVE:
            return data[shard_id::num_shards]


def _load_file_list(root_path: pathlib.Path,
                    file_list_path: Union[pathlib.Path, str, List[str]]) -> List[str]:
    if isinstance(file_list_path, pathlib.Path) or isinstance(file_list_path, str):
        with open(root_path / file_list_path, "r") as input_file:
            file_list_path = input_file.readlines()
    return [x.strip() for x in sorted(file_list_path)]


def get_dali_pipeline(file_root: pathlib.Path,
                      data_file_list: List[str],
                      label_file_list: List[str], *,
                      dont_use_mmap: bool = True,
                      num_shards: int = 1,
                      shard_id: int = 0,
                      apply_log: bool = False,
                      batch_size: int = 1,
                      dali_threads: int = 1,
                      device_id: int = 0,
                      shuffle: bool = False,
                      data_layout: Literal["NDHWC", "NCDHW"] = "NDHWC",
                      sample_shape: List[int] = [128, 128, 128, 4],
                      target_shape: List[int] = [4],
                      seed: int = 1) -> Pipeline:
    SAMPLE_SIZE_DATA = 4*math.prod(sample_shape)
    SAMPLE_SIZE_LABEL = 4*math.prod(target_shape)

    pipeline = Pipeline(batch_size=batch_size,
                        num_threads=dali_threads,
                        device_id=device_id)
    with pipeline:
        numpy_reader = dali_fn.readers.numpy(bytes_per_sample_hint=SAMPLE_SIZE_DATA / 2,
                                             dont_use_mmap=dont_use_mmap,
                                             file_root=str(file_root),
                                             files=data_file_list,
                                             num_shards=num_shards,
                                             shard_id=shard_id,
                                             stick_to_shard=not shuffle,
                                             shuffle_after_epoch=shuffle,
                                             name="data_reader",
                                             seed=seed)
        label_reader = dali_fn.readers.numpy(bytes_per_sample_hint=SAMPLE_SIZE_LABEL,
                                             dont_use_mmap=dont_use_mmap,
                                             file_root=str(file_root),
                                             files=label_file_list,
                                             num_shards=num_shards,
                                             shard_id=shard_id,
                                             stick_to_shard=not shuffle,
                                             shuffle_after_epoch=shuffle,
                                             name="label_reader",
                                             seed=seed)

        feature_map = dali_fn.cast(numpy_reader.gpu(), dtype=dali_types.FLOAT,
                                   bytes_per_sample_hint=SAMPLE_SIZE_DATA)
        if apply_log:
            feature_map = dali_math.log(feature_map + 1.0)
        else:
            feature_map = feature_map / dali_fn.reductions.mean(feature_map)
        if data_layout == "NCDHW":
            feature_map = dali_fn.transpose(feature_map, perm=[3, 0, 1, 2])
        pipeline.set_outputs(feature_map, label_reader.gpu())
    pipeline.build()
    return pipeline

ShardType = Literal["local", "global", "none"]

class CosmoDataset(object):
    def __init__(self,
                 data_dir: pathlib.Path,
                 *,
                 dist: utils.DistributedEnvDesc,
                 use_mmap: bool = False,
                 apply_log: bool = True,
                 dali_threads: int = 1,
                 data_layout: Literal["NDHWC", "NCDHW"] = "NDHWC",
                 input_shape: List[int] = [128, 128, 128, 4],
                 target_shape: List[int] = [4],
                 seed: int = 1,
                 spatial: int = 1):
        self.root_dir = data_dir
        self.data_shapes = (input_shape,
                            target_shape)
        self.threads = dali_threads
        self.use_mmap = use_mmap
        self.apply_log = apply_log
        self.dist = dist
        self.data_layout = data_layout
        self.seed = seed
        self.spatial = spatial

        self.samples_per_file = 1

    def training_dataset(self, batch_size: int, shard: str = "global", shuffle: bool = False, 
                         preshuffle: bool = False, n_samples: int = -1, shard_mult: int = 1, 
                         prestage: bool = False) \
            -> Tuple[dali_mxnet.DALIGenericIterator, int, int]:
        data_path = self.root_dir / "train"

        pipeline_builder, samples = self._construct_pipeline(data_path, batch_size,
                                                             n_samples=n_samples, 
                                                             shard=shard, shuffle=shuffle, 
                                                             prestage=(shard == "local") and prestage,
                                                             preshuffle=preshuffle,
                                                             shard_mult=shard_mult)
        assert samples % self.dist.size == 0, \
            f"Cannot divide {samples} items into {self.dist.size} workers"

        iter_count = samples // (self.dist.size // self.spatial) // batch_size
        
        def iterator_builder():
            pipeline = pipeline_builder()
            iterator = dali_mxnet.DALIGluonIterator(pipeline,
                                                    reader_name="data_reader",
                                                    last_batch_policy=LastBatchPolicy.PARTIAL)
            return iterator

        return iterator_builder, iter_count, samples

    def validation_dataset(self, batch_size: int, shard: bool = True, n_samples: int = -1) \
            -> Tuple[dali_mxnet.DALIGenericIterator, int, int]:
        data_path = self.root_dir / "validation"

        pipeline_builder, samples = self._construct_pipeline(data_path, batch_size,
                                                             n_samples=n_samples, shard="global", 
                                                             prestage=False)
        assert samples % self.dist.size == 0 or not shard, \
            f"Cannot divide {samples} items into {self.dist.size} workers"

        iter_count = samples // ((self.dist.size // self.spatial) if shard else 1) // batch_size

        def iterator_builder():
            pipeline = pipeline_builder()
            iterator = dali_mxnet.DALIGluonIterator(pipeline,
                                                    reader_name="data_reader",
                                                    last_batch_policy=LastBatchPolicy.PARTIAL)
            return iterator

        return iterator_builder, iter_count, samples

    def _construct_pipeline(self, data_dir: pathlib.Path,
                            batch_size: int,
                            n_samples: int = -1,
                            prestage: bool = True,
                            shard: ShardType = "none",
                            shuffle: bool = False,
                            preshuffle: bool = False,
                            shard_mult: int = 1) -> Tuple[Pipeline, int]:

        data_filenames = _load_file_list(data_dir, "files_data.lst")
        label_filenames = _load_file_list(data_dir, "files_label.lst")

        if n_samples > 0:
            data_filenames = data_filenames[:n_samples]
            label_filenames = label_filenames[:n_samples]
        n_samples = len(data_filenames) * self.samples_per_file

        if preshuffle:
            preshuffle_permutation = np.ascontiguousarray(
                np.random.permutation(n_samples))
            self.dist.comm.Bcast(preshuffle_permutation, root=0)

            data_filenames = list(np.array(data_filenames)[preshuffle_permutation])
            label_filenames = list(np.array(label_filenames)[preshuffle_permutation])

        shard_id, num_shards, data_filenames, label_filenames = calculate_sharding(
            self.dist, data_filenames, label_filenames, shard, shard_mult, self.spatial)

        def pipeline_builder():
            if prestage:
                output_path = pathlib.Path("/staging_area", "dataset") / data_dir.parts[-1]
                stage_files(self.dist, data_dir, output_path, 
                            data_filenames, label_filenames)


            return get_dali_pipeline(output_path if prestage else data_dir,
                                     data_filenames,
                                     label_filenames,
                                     dont_use_mmap=not self.use_mmap,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     apply_log=self.apply_log,
                                     batch_size=batch_size,
                                     dali_threads=self.threads,
                                     device_id=self.dist.local_rank,
                                     shuffle=shuffle,
                                     data_layout=self.data_layout,
                                     sample_shape=self.data_shapes[0],
                                     target_shape=self.data_shapes[1],
                                     seed=self.seed)

        return (pipeline_builder,
                n_samples)

def calculate_sharding(dist_desc: utils.DistributedEnvDesc,
                       data_filenames: List[str],
                       label_filenemes: List[str],
                       shard: str, 
                       multiplier: int = 1,
                       spatial: int = 1) -> Tuple[int, int, List[str], List[str]]:
    if shard == "local":
        if multiplier == 1:
            shard_id, num_shards = dist_desc.local_rank // spatial, dist_desc.local_size // spatial
        else:
            node_in_chunk = dist_desc.node % multiplier
            num_shards = dist_desc.local_size * multiplier // spatial
            shard_id = (node_in_chunk * dist_desc.local_size + dist_desc.local_rank) // spatial
        data_split_chunk_count = dist_desc.size // dist_desc.local_size // multiplier
        data_split_chunk_number = dist_desc.node // multiplier
        files_per_split_chunk = len(data_filenames) // data_split_chunk_count

        data_filenames = data_filenames[data_split_chunk_number * files_per_split_chunk:
                                        (data_split_chunk_number+1) * files_per_split_chunk]
        label_filenemes = label_filenemes[data_split_chunk_number * files_per_split_chunk:
                                          (data_split_chunk_number+1) * files_per_split_chunk]

    elif shard == "global":
        shard_id, num_shards = dist_desc.rank // spatial, dist_desc.size // spatial
    else:
        shard_id, num_shards = 0, 1
    
    return shard_id, num_shards, data_filenames, label_filenemes

def stage_files(dist_desc: utils.DistributedEnvDesc,
                data_dir: pathlib.Path,
                output_dir: pathlib.Path,
                data_filenames: List[str],
                label_filenames: List[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)

    import shutil

    copied_files = 0
    for data, label in zip(data_filenames[dist_desc.local_rank::dist_desc.local_size],
                           label_filenames[dist_desc.local_rank::dist_desc.local_size]):
        shutil.copy(data_dir / data, output_dir / data)
        shutil.copy(data_dir / label, output_dir / label)
        copied_files += 1

    #print(f"Node {current_node}, process {dist_desc.local_rank}, dataset contains {len(data_filenames)} samples, ",
    #      f"per node {len(per_node_data_filenames)}, copied {copied_files}", flush=True)

    dist_desc.comm.Barrier()

def get_rec_iterators(args: argparse.Namespace, dist_desc: utils.DistributedEnvDesc) \
        -> Tuple[Callable, int, int]:
    cosmoflow_dataset = CosmoDataset(args.data_root_dir, 
                                     dist=dist_desc,
                                     use_mmap=args.dali_use_mmap,
                                     apply_log=args.apply_log_transform,
                                     dali_threads=args.dali_num_threads,
                                     data_layout=args.data_layout,
                                     seed=args.seed,
                                     spatial=args.spatial_span)
    train_iterator_builder, training_steps, training_samples = cosmoflow_dataset.training_dataset(
        args.training_batch_size, args.shard_type, args.shuffle, args.preshuffle, args.training_samples,
        args.data_shard_multiplier, args.prestage)
    val_iterator_builder, val_steps, val_samples = cosmoflow_dataset.validation_dataset(
        args.validation_batch_size, True, args.validation_samples)
    

    # MLPerf logging of batch size, and number of samples used in training
    utils.logger.event(key=utils.logger.constants.GLOBAL_BATCH_SIZE,
                       value=args.training_batch_size*dist_desc.size // args.spatial_span)
    utils.logger.event(key=utils.logger.constants.TRAIN_SAMPLES, value=training_samples)
    utils.logger.event(key=utils.logger.constants.EVAL_SAMPLES, value=val_samples)

    def iterator_builder():
        staging_start = time.time()
        utils.logger.start(key='staging_start')
        train_iterator = train_iterator_builder()
        val_iterator = val_iterator_builder()
        utils.logger.end(key='staging_stop', metadata={
            'staging_duration': time.time() - staging_start})

        return train_iterator, val_iterator

    return (iterator_builder,
            training_steps,
            val_steps)
