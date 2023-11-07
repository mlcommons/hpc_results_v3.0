# Copyright 2021 DeepMind Technologies Limited
# Copyright 2022 AlQuraishi Laboratory
# Copyright 2023 NVIDIA CORPORATION
#
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
import torch.nn.functional as F

import openfold.model.inductor as inductor
from openfold.model.layer_norm import LayerNorm
from openfold.model.linear import Linear


class SingleTransition(nn.Module):
    """Single Transition module.

    Supplementary '1.8 Structure module': Algorithm 20, lines 8-9.

    Args:
        c_s: Single representation dimension (channels).
        dropout_rate: Dropout rate.

    """

    def __init__(
        self,
        c_s: int,
        dropout_rate: float,
    ) -> None:
        super(SingleTransition, self).__init__()
        self.linear_1 = Linear(c_s, c_s, bias=True, init="relu")
        self.linear_2 = Linear(c_s, c_s, bias=True, init="relu")
        self.linear_3 = Linear(c_s, c_s, bias=True, init="final")
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm(c_s)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if inductor.is_enabled():
            forward_fn = _forward_jit
        else:
            forward_fn = _forward_eager
        s = forward_fn(
            s,
            self.linear_1.weight,
            self.linear_1.bias,
            self.linear_2.weight,
            self.linear_2.bias,
            self.linear_3.weight,
            self.linear_3.bias,
        )
        s = self.layer_norm(self.dropout(s))
        return s


def _forward_eager(
    s: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
) -> torch.Tensor:
    x = F.linear(s, w1, b1)
    x = torch.relu(x)
    x = F.linear(x, w2, b2)
    x = torch.relu(x)
    x = F.linear(x, w3, b3)
    s = s + x
    return s


_forward_jit = torch.compile(_forward_eager)
