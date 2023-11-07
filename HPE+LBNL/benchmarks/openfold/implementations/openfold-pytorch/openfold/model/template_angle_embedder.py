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

import openfold.dynamic_axial_parallelism as dap
import openfold.model.inductor as inductor
from openfold.model.linear import Linear


class TemplateAngleEmbedder(nn.Module):
    """Template Angle Embedder module.

    Embeds the "template_angle_feat" feature.

    Supplementary '1.4 AlphaFold Inference': Algorithm 2, line 7.

    Args:
        ta_dim: Input `template_angle_feat` dimension (channels).
        c_m: Output MSA representation dimension (channels).

    """

    def __init__(
        self,
        ta_dim: int,
        c_m: int,
    ) -> None:
        super(TemplateAngleEmbedder, self).__init__()
        self.linear_1 = Linear(ta_dim, c_m, bias=True, init="relu")
        self.linear_2 = Linear(c_m, c_m, bias=True, init="relu")

    def forward(
        self,
        template_angle_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Template Angle Embedder forward pass.

        Args:
            template_angle_feat: [batch, N_templ, N_res, ta_dim]

        Returns:
            template_angle_embedding: [batch, N_templ, N_res, c_m]

        """
        # dap1 fusion regression
        if inductor.is_enabled_on_hopper() and dap.size() >= 2:
            forward_fn = _forward_jit
        elif inductor.is_enabled_and_autograd_off():
            forward_fn = _forward_jit
        else:
            forward_fn = _forward_eager
        return forward_fn(
            template_angle_feat,
            self.linear_1.weight,
            self.linear_1.bias,
            self.linear_2.weight,
            self.linear_2.bias,
        )


def _forward_eager(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    x = F.linear(x, w1, b1)
    x = torch.relu(x)
    x = F.linear(x, w2, b2)
    return x


_forward_jit = torch.compile(_forward_eager)
