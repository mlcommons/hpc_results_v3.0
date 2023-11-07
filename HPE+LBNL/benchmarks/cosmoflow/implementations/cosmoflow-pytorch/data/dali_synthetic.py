# Copyright (c) 2021-2022 NVIDIA CORPORATION. All rights reserved.
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

from utils import DistributedEnv
from data.dali_core import InputPipelineCore
from omegaconf import DictConfig


import nvidia.dali.fn as dali_fn
import nvidia.dali.math as dali_math
import nvidia.dali.types as dali_types


class SyntheticDataPipeline(InputPipelineCore):
    def __init__(self,
                 config: DictConfig,
                 distenv: DistributedEnv,
                 sample_count: int,
                 device: int):
        super().__init__(config, distenv, sample_count, device)

    def get_inputs(self, pipeline, **kwargs):
        input_data = dali_fn.random.normal(dtype=dali_types.INT16,
                                           shape=list(
                                               self._config["sample_shape"]),
                                           stddev=256, device="gpu")
        input_label = dali_fn.random.normal(dtype=dali_types.FLOAT,
                                            shape=list(
                                                self._config["target_shape"]),
                                            stddev=1.0, device="gpu")

        input_data = dali_math.clamp(input_data, 0, 255)
        input_label = dali_math.clamp(input_label, -0.9, 0.9)

        return input_data, input_label
