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

from omegaconf import DictConfig


from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as dali_fn
import nvidia.dali.math as dali_math
import nvidia.dali.types as dali_types

import nvidia.dali.plugin.pytorch as dali_pytorch

from utils import DistributedEnv

import torch

from typing import Tuple


class InputPipelineCore(object):
    def __init__(self,
                 data_cfg: DictConfig,
                 distenv: DistributedEnv,
                 sample_count: int,
                 device: int,
                 build_now: bool = True,
                 threads: int = None):
        self._config = data_cfg
        self._distenv = distenv

        self._sample_count = sample_count
        self._device = device
        self._threads = threads if threads else self._config["dali_threads"]

        if build_now:
            self._build()

    def _build(self):
        self._pipeline = self._build_pipeline()

    def __len__(self):
        if hasattr(self, "_total_sample_count"):
            return self._total_sample_count
        return self._sample_count

    def __iter__(self):
        class InputMultiplierIter(object):
            def __init__(self,
                         pipeline,
                         sample_count: int,
                         batch_size: int):
                self._dali_iterator = dali_pytorch.DALIGenericIterator(
                    pipeline,
                    ["data", "label"],
                    size=sample_count,
                    auto_reset=True)
                self._buffer = None
                self._batch_size = batch_size
                self._count = 0

            def __iter__(self):
                return self

            def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
                if self._buffer == None or self._count >= self._batch_size:
                    self._count = 0
                    self._buffer = next(self._dali_iterator)
                data, label = self._buffer[0]["data"], self._buffer[0]["label"]
                data = data[self._count:self._count+1]
                label = label[self._count:self._count+1]
                self._count += 1
                return data, label

        return InputMultiplierIter(self._pipeline,
                                   self._sample_count,
                                   self._threads)

    def _build_pipeline(self, **kwargs):
        pipeline = Pipeline(batch_size=self._threads,
                            num_threads=self._threads,
                            device_id=self._device,
                            prefetch_queue_depth={"cpu_size": 4, "gpu_size": 2})
        with pipeline:
            input_data, input_label = self.get_inputs(pipeline=pipeline,
                                                      **kwargs)
            self.get_pipeline(pipeline=pipeline,
                              input_data=input_data,
                              input_label=input_label)
        return pipeline

    def get_pipeline(self,
                     pipeline,
                     input_data,
                     input_label):
        feature_map = input_data.gpu()
        #feature_map = dali_fn.cast(input_data.gpu(), dtype=dali_types.FLOAT)
        # if self._config["apply_log"]:
        #    feature_map = dali_math.log(feature_map + 1.0)
        # else:
        #    feature_map = feature_map / dali_fn.reductions.mean(feature_map)

        pipeline.set_outputs(feature_map, input_label.gpu())

    def get_inputs(self, pipelin, **kwargs):
        pass

    def stage_data(self, executor=None, profile: bool = False):
        def no_wait():
            pass
        return no_wait
