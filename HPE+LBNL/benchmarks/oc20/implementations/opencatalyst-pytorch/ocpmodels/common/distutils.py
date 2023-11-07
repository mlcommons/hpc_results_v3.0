"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import ctypes
import os
from socket import gethostname

import numpy as np
import torch
import torch.distributed as dist
from mpi4py import MPI


def setup(config):
    mpi_comm = MPI.COMM_WORLD
    per_instance_comm = mpi_comm
    world_size = mpi_comm.Get_size()
    config["nproc_per_node"] = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
    config["train_ranks"] = list(range(world_size))
    config["eval_ranks"] = list(range(world_size))
    config["global_rank"] = mpi_comm.Get_rank()
    config["local_rank"] = mpi_comm.Get_rank() % config["nproc_per_node"]

    num_instances, instance = config["instances"], 0
    if num_instances > 1:  # weak scaling
        processes_per_instance = mpi_comm.Get_size() // num_instances
        assert (
            mpi_comm.Get_size() % num_instances == 0
        ), f"Cannot split {mpi_comm.Get_size()} processes into {num_instances} instancess"

        instance = mpi_comm.Get_rank() // processes_per_instance
        per_instance_comm = mpi_comm.Split(color=instance, key=mpi_comm.Get_rank())
    elif config["nodes_for_eval"] > 0:  # async eval
        train_ranks, eval_ranks, transfer_ranks = assign_mpiranks(
            config["local_rank"], world_size, config["nodes_for_eval"], config["nproc_per_node"]
        )
        instance = 0 if mpi_comm.Get_rank() in train_ranks else 1
        per_instance_comm = mpi_comm.Split(color=instance, key=mpi_comm.Get_rank())
        transfer_comm = get_group_comm(mpi_comm, transfer_ranks)
        config["train_ranks"] = train_ranks
        config["eval_ranks"] = eval_ranks
        config["transfer_ranks"] = transfer_ranks
        config["transfer_comm"] = transfer_comm

    init_method = f"tcp://{per_instance_comm.allgather(gethostname())[0]}:12345"
    config["master_mpi_comm"] = mpi_comm
    config["instance_mpi_comm"] = per_instance_comm
    config["instance"] = instance
    config["world_size"] = per_instance_comm.Get_size()
    config["rank"] = per_instance_comm.Get_rank()

    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=config["rank"],
        world_size=config["world_size"],
    )
    torch.cuda.set_device(config["local_rank"])

    _libcudart = ctypes.CDLL("libcudart.so")
    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128


def local_barrier(config):
    config["instance_mpi_comm"].Barrier()


def global_barrier(config):
    config["master_mpi_comm"].Barrier()


def cleanup():
    dist.destroy_process_group()


def initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if initialized() else 0


def get_world_size():
    return dist.get_world_size() if initialized() else 1


def is_master():
    return get_rank() == 0


def is_eval_master(config):
    if config["instances"] > 1:
        return get_rank() == 0
    return config["global_rank"] == config["eval_ranks"][0]


def synchronize():
    if get_world_size() == 1:
        return
    dist.barrier()


def broadcast(tensor, src, group=dist.group.WORLD, async_op=False):
    if get_world_size() == 1:
        return
    dist.broadcast(tensor, src, group, async_op)


def all_reduce(data, group=dist.group.WORLD, average=False, device=None):
    if get_world_size() == 1:
        return data
    tensor = data
    if not isinstance(data, torch.Tensor):
        tensor = torch.tensor(data)
    if device is not None:
        tensor = tensor.cuda(device)
    dist.all_reduce(tensor, group=group)
    if average:
        tensor /= get_world_size()
    if not isinstance(data, torch.Tensor):
        result = tensor.cpu().numpy() if tensor.numel() > 1 else tensor.item()
    else:
        result = tensor
    return result


def all_gather(data, group=dist.group.WORLD, device=None):
    if get_world_size() == 1:
        return data
    tensor = data
    if not isinstance(data, torch.Tensor):
        tensor = torch.tensor(data)
    if device is not None:
        tensor = tensor.cuda(device)
    tensor_list = [tensor.new_zeros(tensor.shape) for _ in range(group.size())]
    dist.all_gather(tensor_list, tensor, group=group)
    if not isinstance(data, torch.Tensor):
        result = [tensor.cpu().numpy() for tensor in tensor_list]
    else:
        result = tensor_list
    return result


def create_groups(group_size):
    groups = []
    num_groups = get_world_size() // group_size
    group_id = get_rank() // group_size
    for i in range(num_groups):
        groups.append(dist.new_group(ranks=list(range(i * group_size, (i + 1) * group_size))))
    return groups, group_id


def assign_mpiranks(local_rank, size, nodes_for_eval, gpu_per_node):
    total_ranks = list(range(size))
    train_ranks = total_ranks[: size - nodes_for_eval * gpu_per_node]
    eval_ranks = total_ranks[size - nodes_for_eval * gpu_per_node :]
    transfer_ranks = [train_ranks[local_rank], *[x for x in eval_ranks if x % gpu_per_node == local_rank]]
    return train_ranks, eval_ranks, transfer_ranks


def get_group_comm(comm, ranks):
    xcomm = None
    if ranks:
        xgroup = comm.group.Incl(ranks)
        xcomm = comm.Create_group(xgroup)
    return xcomm


def sync_training_and_evaluation(config, model, stop_training):
    local_stop_training = np.array([stop_training], dtype=np.int32)
    global_stop_training = np.zeros(1, dtype=np.int32)
    config["master_mpi_comm"].Allreduce(local_stop_training, global_stop_training, MPI.SUM)

    if config["global_rank"] in config["transfer_ranks"]:
        broadcast_model(model, config)

    # Evaluation found end of training
    if global_stop_training != 0:
        stop_training = True

    return stop_training, model


def broadcast_model(model, config):
    result, irequests = {}, []
    comm, rank, eval_ranks = config["transfer_comm"], config["global_rank"], config["eval_ranks"]

    for name, param in model.named_parameters():
        result[name] = param.data.cpu().numpy()
        irequests.append(comm.Ibcast([result[name], result[name].size * result[name].itemsize, MPI.CHAR], root=0))
    MPI.Request.waitall(irequests)

    if rank in eval_ranks:
        device = f"""cuda:{config["local_rank"]}"""
        for name, param in model.named_parameters():
            param.data = torch.tensor(result[name], device=device)
