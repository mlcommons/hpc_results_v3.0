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

from __future__ import annotations

from typing import Tuple

import torch

import openfold.distributed as dist

# whether DAP has been initialized or not
_DAP_INITIALIZED = False

# whether DAP is enabled or not
_DAP_ENABLED = None

# DAP size, one of: 0, 1, 2, 4 or 8
_DAP_SIZE = 0

# DAP process group
_DAP_GROUP = None

# DAP group rank: from 0 to num_dap_groups-1
_DAP_GROUP_RANK = None

# process rank inside DAP group: from 0 to dap_size-1
_DAP_RANK = None


def initialize(dap_size: int) -> None:
    """
    Initialize Dynamic Axial Parallelism (DAP).

    Args:
        dap_size: number of GPUs used in DAP group.

    """
    global _DAP_INITIALIZED
    global _DAP_ENABLED
    global _DAP_SIZE
    global _DAP_GROUP
    global _DAP_GROUP_RANK
    global _DAP_RANK

    assert not _DAP_INITIALIZED
    assert _DAP_ENABLED is None
    assert _DAP_SIZE == 0
    assert _DAP_GROUP is None
    assert _DAP_GROUP_RANK is None
    assert _DAP_RANK is None
    assert dap_size in {1, 2, 4, 8}
    assert dist.is_initialized()

    num_train_ranks = len(dist.train_ranks())
    if num_train_ranks % dap_size != 0:
        raise RuntimeError(
            f"num_train_ranks={num_train_ranks} is not divisible by dap_size={dap_size}"
        )
    num_dap_groups = num_train_ranks // dap_size

    for dap_group_rank in range(num_dap_groups):
        ranks_forming_dap_group = list(
            range(
                dap_group_rank * dap_size,
                (dap_group_rank + 1) * dap_size,
            ),
        )
        group = torch.distributed.new_group(ranks_forming_dap_group)
        if dist.rank() in ranks_forming_dap_group:
            _DAP_GROUP = group
            assert dap_group_rank == dist.rank() // dap_size
            _DAP_GROUP_RANK = dap_group_rank
            _DAP_RANK = dist.rank() % dap_size

    _DAP_SIZE = dap_size
    _DAP_ENABLED = True
    _DAP_INITIALIZED = True


def is_initialized() -> bool:
    return _DAP_INITIALIZED


def is_enabled() -> bool:
    return bool(_DAP_ENABLED)


def size() -> int:
    return _DAP_SIZE


def group() -> torch.distributed.ProcessGroup:
    assert _DAP_INITIALIZED
    return _DAP_GROUP


def group_rank() -> int:
    assert _DAP_INITIALIZED
    return _DAP_GROUP_RANK


def rank() -> int:
    assert _DAP_INITIALIZED
    return _DAP_RANK


def _enable() -> None:
    global _DAP_ENABLED
    _DAP_ENABLED = True


def _disable() -> None:
    global _DAP_ENABLED
    _DAP_ENABLED = False


class Enable(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Enable) -> None:
        return _enable()

    @staticmethod
    def backward(ctx: Enable) -> None:
        return _disable()


class Disable(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Disable) -> None:
        return _disable()

    @staticmethod
    def backward(ctx: Disable) -> None:
        return _enable()


def enable() -> None:
    if is_initialized():
        if torch.is_grad_enabled():
            Enable.apply()
        else:
            _enable()


def disable() -> None:
    if is_initialized():
        if torch.is_grad_enabled():
            Disable.apply()
        else:
            _disable()


def _split(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    assert tensor.size(dim) % _DAP_SIZE == 0
    chunks = tensor.chunk(_DAP_SIZE, dim=dim)
    output = chunks[_DAP_RANK]
    return output


def _gather(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    tensor = tensor.contiguous()

    if dim == 1 and tensor.size(0) == 1:
        # special case: tensors in the list are contiguous parts of the output
        output_shape = list(tensor.shape)
        output_shape[1] *= _DAP_SIZE
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        tensor_list = list(output.chunk(_DAP_SIZE, dim=1))
        torch.distributed.all_gather(
            tensor_list=tensor_list,
            tensor=tensor,
            group=_DAP_GROUP,
            async_op=False,
        )
    else:
        # tensors in the list are NOT contiguous parts of the output
        tensor_list = [torch.empty_like(tensor) for _ in range(_DAP_SIZE)]
        torch.distributed.all_gather(
            tensor_list=tensor_list,
            tensor=tensor,
            group=_DAP_GROUP,
            async_op=False,
        )
        output = torch.cat(tensor_list, dim=dim)

    return output


def _all_reduce_sum_split(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    tensor = tensor.contiguous()

    torch.distributed.all_reduce(
        tensor=tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=_DAP_GROUP,
    )

    assert tensor.size(dim) % _DAP_SIZE == 0
    chunks = tensor.chunk(_DAP_SIZE, dim=dim)
    output = chunks[_DAP_RANK]

    return output


class Scatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Scatter, input: torch.Tensor, dim: int) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _split(input, dim=dim)

    @staticmethod
    def backward(ctx: Scatter, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return _gather(grad_output, dim=int(ctx.saved_tensors[0][0])), None


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Gather, input: torch.Tensor, dim: int) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _gather(input, dim=dim)

    @staticmethod
    def backward(ctx: Gather, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return _split(grad_output, dim=int(ctx.saved_tensors[0][0])), None


class GatherAllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx: GatherAllReduceSum, input: torch.Tensor, dim: int) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _gather(input, dim=dim)

    @staticmethod
    def backward(
        ctx: GatherAllReduceSum, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        return (
            _all_reduce_sum_split(grad_output, dim=int(ctx.saved_tensors[0][0])),
            None,
        )


def scatter(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    if not _DAP_ENABLED:
        return tensor

    if torch.is_grad_enabled() and tensor.requires_grad:
        tensor = Scatter.apply(tensor, dim)
    else:
        tensor = _split(tensor, dim=dim)

    return tensor


def gather(tensor: torch.Tensor, dim: int, bwd: str = "split") -> torch.Tensor:
    if not _DAP_ENABLED:
        return tensor

    if torch.is_grad_enabled() and tensor.requires_grad:
        if bwd == "split":
            tensor = Gather.apply(tensor, dim)
        elif bwd == "all_reduce_sum_split":
            tensor = GatherAllReduceSum.apply(tensor, dim)
        else:
            raise ValueError(f"unknown bwd={repr(bwd)}")
    else:
        tensor = _gather(tensor, dim=dim)

    return tensor


def _all_to_all(tensor: torch.Tensor, in_dim: int, out_dim: int) -> torch.Tensor:
    assert tensor.size(in_dim) % _DAP_SIZE == 0

    input_tensor_list = [
        input_tensor.contiguous()
        for input_tensor in tensor.chunk(_DAP_SIZE, dim=in_dim)
    ]

    if out_dim == 1 and tensor.size(0) == 1:
        # special case: output tensors in the list are contiguous parts of the output
        output_shape = list(input_tensor_list[0].shape)
        output_shape[1] *= _DAP_SIZE
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        output_tensor_list = list(output.chunk(_DAP_SIZE, dim=1))
        torch.distributed.all_to_all(
            output_tensor_list=output_tensor_list,
            input_tensor_list=input_tensor_list,
            group=_DAP_GROUP,
            async_op=False,
        )
    else:
        # output tensors in the list are NOT contiguous parts of the output
        output_tensor_list = [
            torch.empty_like(input_tensor) for input_tensor in input_tensor_list
        ]
        torch.distributed.all_to_all(
            output_tensor_list=output_tensor_list,
            input_tensor_list=input_tensor_list,
            group=_DAP_GROUP,
            async_op=False,
        )
        output = torch.cat(output_tensor_list, dim=out_dim)

    return output


class All_to_All(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: All_to_All, input: torch.Tensor, in_dim: int, out_dim: int
    ) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor([in_dim, out_dim]))
        return _all_to_all(input, in_dim=in_dim, out_dim=out_dim)

    @staticmethod
    def backward(
        ctx: All_to_All, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None]:
        return (
            _all_to_all(
                grad_output,
                in_dim=int(ctx.saved_tensors[0][1]),
                out_dim=int(ctx.saved_tensors[0][0]),
            ),
            None,
            None,
        )


def col_to_row(tensor: torch.Tensor) -> torch.Tensor:
    if not _DAP_ENABLED:
        return tensor

    if torch.is_grad_enabled() and tensor.requires_grad:
        tensor = All_to_All.apply(tensor, 1, 2)
    else:
        tensor = _all_to_all(tensor, in_dim=1, out_dim=2)

    return tensor


def row_to_col(tensor: torch.Tensor) -> torch.Tensor:
    if not _DAP_ENABLED:
        return tensor

    if torch.is_grad_enabled() and tensor.requires_grad:
        tensor = All_to_All.apply(tensor, 2, 1)
    else:
        tensor = _all_to_all(tensor, in_dim=2, out_dim=1)

    return tensor
