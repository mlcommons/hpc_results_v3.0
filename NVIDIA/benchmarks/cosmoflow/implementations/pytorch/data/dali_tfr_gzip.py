# Copyright (c) 2021-2023 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from utils.executor import AbstractStagerExecutor
from utils import DistributedEnv
from data.dali_core import InputPipelineCore

from omegaconf import DictConfig
from typing import Literal, List, Optional, Tuple
from itertools import chain


import nvidia.dali.fn as dali_fn
import nvidia.dali.tfrecord as dali_tfrec
import nvidia.dali.types as dali_types

import numpy as np

import pathlib
import random
import os


class TFRecordDataPipeline(InputPipelineCore):
    def __init__(self,
                 config: DictConfig,
                 distenv: DistributedEnv,
                 mode: Literal["train", "validation"],
                 sample_count: int,
                 device: int,
                 seed: Optional[int] = None,
                 threads: Optional[int] = None):
        super().__init__(config, distenv, sample_count, device, 
                         build_now=False, threads=threads)

        self._file_root_path = os.path.join(config["root_dir"], mode)
        self._file_staged_path = pathlib.Path(os.path.join(
            config["stage"], mode) if config["stage"] != False else self._file_root_path)

        self._shuffle = config["shuffle"] and mode == "train"
        if "use_direct_io" not in config:
            self._use_direct_io = True
        else:
            self._use_direct_io = config["use_direct_io"]

        self._seed = seed if seed else random.randint(0, 65536)
        self._mode = mode

        self._setup_sharding()
        self._prepare_file_list()
        self._build()

    def get_inputs(self, pipeline, **kwargs):
        sample_size_data = np.prod(self._config["sample_shape"])
        sample_size_label = np.prod(self._config["target_shape"])
        tf_reader = dali_fn.readers.tfrecord(features={"x": dali_tfrec.FixedLenFeature((), dali_tfrec.string, ""),
                                                       "y": dali_tfrec.FixedLenFeature((4,), dali_tfrec.float32, 0.0)},
                                             path=[str(self._file_staged_path / sample)
                                                   for sample in self._file_list_data],
                                             index_path=[str(self._file_staged_path / f"{sample}.idx")
                                                         for sample in self._file_list_data],
                                             bytes_per_sample_hint=sample_size_data + sample_size_label,
                                             prefetch_queue_depth=64,
                                             use_o_direct=self._use_direct_io,
                                             dont_use_mmap=self._use_direct_io,
                                             num_shards=self._num_shards,
                                             shard_id=self._shard_id,
                                             stick_to_shard=True,
                                             random_shuffle=self._shuffle,
                                             name="data_reader",
                                             seed=self._seed)
        data = dali_fn.reinterpret(tf_reader["x"].gpu(),
                                   dtype=dali_types.INT16,
                                   shape=[128, 128, 128, 4])
        return data, tf_reader["y"]

    def _setup_sharding(self):
        if self._config["shard_type"] == "global":
            self._shard_id = self._distenv.rank
            self._num_shards = self._distenv.size
            self._master_shard_id = 0
            self._num_master_shard = 1
        elif self._config["shard_type"] == "local":
            number_of_nodes = self._distenv.size // self._distenv.local_size
            data_multiplier = self._config["shard_multiplier"]

            self._num_master_shard = number_of_nodes // data_multiplier
            self._master_shard_id = self._distenv.rank // (
                self._distenv.local_size * data_multiplier)

            self._num_shards = self._distenv.local_size * data_multiplier
            self._shard_id = self._distenv.rank % self._num_shards
        else:
            raise RuntimeError(f"Invalid shard type {self._config['shard_type']}. "
                               f"Valid options are: 'global', 'local'")

    def _prepare_file_list(self):
        data_filenames, file_sizes = _load_file_list(
            self._file_root_path, "files_data.lst")

        self._sample_count = len(data_filenames) // self._distenv.size
        self._total_sample_count = len(data_filenames)
        self._samples_per_master_shard = len(
            data_filenames) // self._num_master_shard

        if self._config["preshuffle"] and self._mode == "train":
            preshuffle_permutation = np.ascontiguousarray(
                np.random.permutation(len(data_filenames)))
            self._distenv.master_mpi_comm.Bcast(
                preshuffle_permutation, root=0)

            data_filenames = list(np.array(data_filenames)[
                preshuffle_permutation])
            file_sizes = file_sizes[preshuffle_permutation]

        #self._file_list_data = data_filenames[self._master_shard_id * self._samples_per_master_shard:
        #                                      (self._master_shard_id+1) * self._samples_per_master_shard]
        #self._file_list_size = file_sizes[self._master_shard_id * self._samples_per_master_shard:
        #                                  (self._master_shard_id+1) * self._samples_per_master_shard]
        self._file_list_data = [
            x for i, x in enumerate(data_filenames)
            if i % self._num_master_shard == self._master_shard_id]
        self._file_list_size = [
            x for i, x in enumerate(file_sizes)
            if i % self._num_master_shard == self._master_shard_id]

        # Few assertion for testing correctness of calculations performed here
        assert len(
            self._file_list_data) // self._num_shards == self._sample_count

    def stage_data(self, executor: AbstractStagerExecutor, *, profile: bool = False):
        if self._config["stage"] == False or self._config["shard_type"] != "local":
            return
        os.makedirs(self._file_staged_path, exist_ok=True)
        future = executor.stage_files(self._file_list_data[self._distenv.local_rank::self._distenv.local_size],
                                      None,
                                      pathlib.Path(self._file_root_path),
                                      pathlib.Path(self._file_staged_path),
                                      profile=profile,
                                      sizes=self._file_list_size[self._distenv.local_rank::self._distenv.local_size],
                                      compressed=True)
        return future

    def check_correctness(self):
        import glob
        file_list_all = [os.path.basename(x) for x in glob.glob(
            f"{str(self._file_staged_path)}/*")]
        file_list_data = set(
            [x for x in file_list_all if not x.endswith(".idx")])
        file_list_idx = set([x for x in file_list_all if x.endswith(".idx")])

        print(
            f"[{self._distenv.instance}, {self._distenv.rank}, {self._distenv.local_rank}]", file_list_all)

        for expected_file in self._file_list_data:
            if expected_file not in file_list_data:
                print(f"[{self._distenv.instance}, {self._distenv.rank}, {self._distenv.local_rank}]"
                      f"File {expected_file} expected to be staged, but not found.")
            if f"{expected_file}.idx" not in file_list_idx:
                print(f"[{self._distenv.instance}, {self._distenv.rank}, {self._distenv.local_rank}]"
                      f"File {expected_file} expected to have idx file, but not found.")

    @staticmethod
    def build(config: DictConfig,
              distenv: DistributedEnv,
              device: int,
              seed: Optional[int] = None,
              **kwargs) -> Tuple[InputPipelineCore, InputPipelineCore]:

        train_args = {k[6:]: v for k,
                      v in chain(config.items(), kwargs.items()) 
                      if k.startswith("train_")}
        val_args = {k[4:]: v for k,
                    v in chain(config.items(), kwargs.items()) 
                    if k.startswith("val_")}

        default_dict = {"device": device,
                        "seed": seed}

        train_args.update(default_dict)
        train_args.setdefault("sample_count", -1)

        val_args.update(default_dict)
        val_args.setdefault("sample_count", -1)

        return (TFRecordDataPipeline(config, distenv, "train",
                                     **train_args),
                TFRecordDataPipeline(config, distenv, "validation",
                                     **val_args))


def _load_file_list(root_path: str,
                    file_list_path: str) -> Tuple[List[str], np.ndarray]:
    with open(os.path.join(root_path, file_list_path), "r") as input_file:
        file_list_path = input_file.readlines()

    result_names = []
    result_size = []
    for line in sorted(file_list_path):
        file_name, size = line.split(" ")

        result_names.append(file_name.strip())
        result_size.append(int(size.strip()))
    return result_names, np.array(result_size)
