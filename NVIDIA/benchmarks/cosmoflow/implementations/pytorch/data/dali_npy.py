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


import nvidia.dali.fn as dali_fn

import numpy as np

import pathlib
import random
import os


class NPyLegacyDataPipeline(InputPipelineCore):
    def __init__(self,
                 config: DictConfig,
                 distenv: DistributedEnv,
                 mode: Literal["train", "validation"],
                 sample_count: int,
                 device: int,
                 seed: Optional[int] = None):
        super().__init__(config, distenv, sample_count, device, build_now=False)

        self._file_root_path = os.path.join(config["root_dir"], mode)
        self._file_staged_path = os.path.join(
            config["stage"], mode) if config["stage"] != False else self._file_root_path

        self._shuffle = config["shuffle"] and mode == "train"
        if hasattr(config, "io_device"):
            self._io_device = config["io_device"]
        else:
            self._io_device = "cpu"
        if self._io_device not in ["cpu", "gpu"]:
            raise NotImplementedError(f"Invalid io_device: {self._io_device}. "
                                      f"Valid options are: 'cpu', 'gpu'")
        if hasattr(config, "disable_mmap"):
            self._disable_mmap = config["disable_mmap"]
        else:
            self._disable_mmap = False
        self._seed = seed if seed else random.randint(0, 65536)
        self._mode = mode

        self._setup_sharding()
        self._prepare_file_list()
        self._build()

    def get_inputs(self, pipeline, **kwargs):
        sample_size_data = np.prod(self._config["sample_shape"])
        sample_size_label = np.prod(self._config["target_shape"])
        numpy_reader = dali_fn.readers.numpy(device=self._io_device,
                                             bytes_per_sample_hint=sample_size_data * 2,
                                             dont_use_mmap=(self._disable_mmap
                                                            or (self._io_device=="gpu")),
                                             cache_header_information=True,
                                             file_root=str(
                                                 self._file_staged_path),
                                             files=self._file_list_data,
                                             num_shards=self._num_shards,
                                             shard_id=self._shard_id,
                                             stick_to_shard=not self._shuffle,
                                             shuffle_after_epoch=self._shuffle,
                                             name="data_reader",
                                             seed=self._seed)
        # always run the labels un the CPU
        label_reader = dali_fn.readers.numpy(device="cpu",
                                             bytes_per_sample_hint=sample_size_label * 4,
                                             dont_use_mmap=self._disable_mmap,
                                             cache_header_information=True,
                                             file_root=str(
                                                 self._file_staged_path),
                                             files=self._file_list_label,
                                             num_shards=self._num_shards,
                                             shard_id=self._shard_id,
                                             stick_to_shard=not self._shuffle,
                                             shuffle_after_epoch=self._shuffle,
                                             name="label_reader",
                                             seed=self._seed)
        return numpy_reader, label_reader

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
        data_filenames = _load_file_list(
            self._file_root_path, "files_data.lst")
        label_filenames = _load_file_list(
            self._file_root_path, "files_label.lst")

        self._sample_count = len(data_filenames) // self._distenv.size
        self._total_sample_count = len(data_filenames)
        self._samples_per_master_shard = len(
            data_filenames) // self._num_master_shard

        if self._config["preshuffle"] and self._mode == "train":
            preshuffle_permutation = np.ascontiguousarray(
                np.random.permutation(len(data_filenames)))
            self._distenv.instance_mpi_comm.Bcast(
                preshuffle_permutation, root=0)

            data_filenames = list(np.array(data_filenames)[
                                  preshuffle_permutation])
            label_filenames = list(np.array(label_filenames)[
                                   preshuffle_permutation])

        self._file_list_data = data_filenames[self._master_shard_id * self._samples_per_master_shard:
                                              (self._master_shard_id+1) * self._samples_per_master_shard]
        self._file_list_label = label_filenames[self._master_shard_id * self._samples_per_master_shard:
                                                (self._master_shard_id+1) * self._samples_per_master_shard]

        # Few assertion for testing correctness of calculations performed here
        assert len(self._file_list_data) // self._num_shards == self._sample_count
        assert len(data_filenames) == len(label_filenames)
        assert len(self._file_list_data) == len(self._file_list_label)

    def stage_data(self, executor: AbstractStagerExecutor, *, profile: bool = False):
        if self._config["stage"] == False or self._config["shard_type"] != "local":
            return

        os.makedirs(self._file_staged_path, exist_ok=True)
        future = executor.stage_files(self._file_list_data[self._distenv.local_rank::self._distenv.local_size],
                                      self._file_list_label[self._distenv.local_rank::self._distenv.local_size],
                                      pathlib.Path(self._file_root_path),
                                      pathlib.Path(self._file_staged_path),
                                      profile=profile)
        return future

    @staticmethod
    def build(config: DictConfig,
              distenv: DistributedEnv,
              device: int,
              seed: Optional[int] = None,
              **kwargs) -> Tuple[InputPipelineCore, InputPipelineCore]:
        train_args = {k[6:]: v for k,
                      v in kwargs.items() if k.startswith("train_")}
        val_args = {k[4:]: v for k,
                    v in kwargs.items() if k.startswith("val_")}
        default_dict = {"device": device,
                        "seed": seed}

        train_args.update(default_dict)
        train_args.setdefault("sample_count", -1)

        val_args.update(default_dict)
        val_args.setdefault("sample_count", -1)

        return (NPyLegacyDataPipeline(config, distenv, "train",
                                      **train_args),
                NPyLegacyDataPipeline(config, distenv, "validation",
                                      **val_args))


def _load_file_list(root_path: str,
                    file_list_path: str) -> List[str]:
    with open(os.path.join(root_path, file_list_path), "r") as input_file:
        file_list_path = input_file.readlines()
    return [x.strip() for x in sorted(file_list_path)]
