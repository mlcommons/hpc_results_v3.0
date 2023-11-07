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
import torch.nn as nn
import torch.nn.functional as nnf

from typing import Iterable, Optional

from model.utils import Convolution3DLayout
from utils import ProfilerSection


class Conv3DActMP(nn.Module):
    def __init__(self,
                 conv_kernel: int,
                 conv_channel_in: int,
                 conv_channel_out: int):
        super().__init__()

        self.conv = nn.Conv3d(conv_channel_in, conv_channel_out,
                              kernel_size=conv_kernel,
                              stride=1, padding=1, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.3)
        self.mp = nn.MaxPool3d(kernel_size=2, stride=2)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mp(self.act(self.conv(x)))


class CosmoFlow(nn.Module):
    def __init__(self,
                 n_conv_layers: int,
                 n_conv_filters: int,
                 conv_kernel: int,
                 dropout_rate: Optional[float] = 0.5):
        super().__init__()

        self.conv_seq = nn.ModuleList()
        input_channel_size = 4
        for i in range(n_conv_layers):
            output_channel_size = n_conv_filters * (1 << i)
            self.conv_seq.append(Conv3DActMP(conv_kernel,
                                             input_channel_size,
                                             output_channel_size))
            input_channel_size = output_channel_size

        flatten_inputs = 128 // (2 ** n_conv_layers)
        flatten_inputs = (flatten_inputs ** 3) * input_channel_size
        self.dense1 = nn.Linear(flatten_inputs, 128)
        self.dense2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 4)

        self.dropout_rate = dropout_rate
        if self.dropout_rate is not None:
            self.dr1 = nn.Dropout(p=self.dropout_rate)
            self.dr2 = nn.Dropout(p=self.dropout_rate)

        for layer in [self.dense1, self.dense2, self.output]:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv_layer in enumerate(self.conv_seq):
            with ProfilerSection(f"conv_{i}", True):
                x = conv_layer(x)
        #x = x.flatten(1)

        # for tf compatibility
        x = x.permute(0, 2, 3, 4, 1).flatten(1)

        with ProfilerSection(f"dense_1", True):
            x = nnf.leaky_relu(self.dense1(x.flatten(1)), negative_slope=0.3)
            if self.dropout_rate is not None:
                x = self.dr1(x)

        with ProfilerSection(f"dense_2", True):
            x = nnf.leaky_relu(self.dense2(x), negative_slope=0.3)
            if self.dropout_rate is not None:
                x = self.dr2(x)

        with ProfilerSection("output", True):
            return torch.tanh(self.output(x)) * 1.2


def get_standard_cosmoflow_model(kernel_size: int = 3,
                                 n_conv_layer: int = 5,
                                 n_conv_filters: int = 32,
                                 dropout_rate: Optional[float] = 0.5,
                                 layout: Convolution3DLayout = Convolution3DLayout.NCDHW,
                                 script: bool = False,
                                 device: str = "cuda") -> nn.Module:
    model = CosmoFlow(n_conv_layers=n_conv_layer,
                      n_conv_filters=n_conv_filters,
                      conv_kernel=kernel_size,
                      dropout_rate=dropout_rate)

    model.to(memory_format=layout.pytorch_memory_format,
             device=device)

    if script:
        model = torch.jit.script(model)
    return model
