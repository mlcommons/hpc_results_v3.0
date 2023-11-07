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

import torch

from typing import Tuple


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self,
                 n_samples: int,
                 sample_shape: Tuple[int, int, int],
                 sample_channels: int):
        self.shape = (sample_channels, *sample_shape)
        self.data = torch.rand((n_samples, *self.shape),
                               dtype=torch.float32)
        self.target = torch.rand((n_samples, 4),
                                 dtype=torch.float32)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.target[idx]
