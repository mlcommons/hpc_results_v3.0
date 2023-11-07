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

import enum

import torch


class Convolution3DLayout(enum.Enum):
    NCDHW = "NCDHW"
    NDHWC = "NDHWC"

    @property
    def channel_last(self) -> bool:
        return self == Convolution3DLayout.NDHWC

    @property
    def pytorch_memory_format(self) -> torch.memory_format:
        return torch.channels_last_3d if self.channel_last else torch.contiguous_format
