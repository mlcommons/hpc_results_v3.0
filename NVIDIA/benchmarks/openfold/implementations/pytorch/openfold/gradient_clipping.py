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

import math
from typing import Optional

import torch


class AsyncGradientClipping:
    def __init__(
        self,
        device: torch.device,
        comm_group: Optional[torch.distributed.ProcessGroup] = None,
        norm_type: float = 2.0,
    ) -> None:
        self.comm_group = (
            comm_group if comm_group is not None else torch.distributed.group.WORLD
        )
        self.norm_type = norm_type
        self._norm_acc = torch.tensor(0.0, device=device)

    def get_clip_scale(self, max_norm: float, eps: float = 1e-6) -> float:
        grad_norm_acc = self._norm_acc.item()
        grad_norm = math.pow(grad_norm_acc + eps, 1.0 / self.norm_type)
        clip_scale = min(max_norm / grad_norm, 1.0)
        self._norm_acc.zero_()
        return clip_scale


def update_norm_from_buckets(
    state: AsyncGradientClipping,
    bucket: torch.distributed.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    grad = bucket.buffer()
    world_size = state.comm_group.size()
    grad.div_(world_size)

    def _acc_grad_norm(fut: torch.futures.Future[torch.Tensor]) -> torch.Tensor:
        synced_grad = fut.value()[0]  # fut.value() type: List[torch.Tensor]
        grad_to_power_p = synced_grad.detach().pow(state.norm_type)
        state._norm_acc += grad_to_power_p.sum()
        return synced_grad

    return (
        torch.distributed.all_reduce(grad, group=state.comm_group, async_op=True)
        .get_future()
        .then(_acc_grad_norm)
    )
