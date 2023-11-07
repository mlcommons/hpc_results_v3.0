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

from mlperf_common.frameworks.pyt import PyTCommunicationHandler
from mlperf_common.logging import MLLoggerWrapper
from typing import Optional, Any
from ctypes import cdll

from mlperf_logging import mllog
from mpi4py import MPI
from omegaconf import DictConfig

from hydra.utils import get_original_cwd

import dataclasses
import time
import os


import torch
import torch.distributed


@dataclasses.dataclass(frozen=True)
class DistributedEnv(object):
    local_rank: int
    local_size: int
    rank: int
    size: int

    master_mpi_comm: Optional[MPI.Comm]
    instance_mpi_comm: Optional[MPI.Comm]
    instance: int
    num_instances: int

    @property
    def master(self) -> bool:
        return self.rank == 0

    @property
    def local_master(self) -> bool:
        return self.local_rank == 0

    @property
    def is_single(self) -> bool:
        return self.size == 1

    def global_barrier(self) -> None:
        if self.master_mpi_comm is not None:
            self.master_mpi_comm.Barrier()

    def local_barrier(self) -> None:
        if self.instance_mpi_comm is not None:
            self.instance_mpi_comm.Barrier()

    @staticmethod
    def create_single() -> "DistributedEnv":
        return DistributedEnv(0, 1, 0, 1, None, None, 0, 1)

    @staticmethod
    def create_from_mpi(config: DictConfig) -> "DistributedEnv":
        mpi_comm = MPI.COMM_WORLD
        per_instance_comm = mpi_comm

        instance = 0
        num_instances = 1
        if "instances" in config:
            num_instances = config["instances"]
            processes_per_instance = mpi_comm.Get_size() // num_instances

            assert mpi_comm.Get_size() % num_instances == 0, \
                f"Cannot split {mpi_comm.Get_size()} processes into {num_instances} instancess"
            instance = mpi_comm.Get_rank() // processes_per_instance

            per_instance_comm = mpi_comm.Split(
                color=instance, key=mpi_comm.Get_rank())

        return DistributedEnv(mpi_comm.Get_rank() % config["mpi"]["local_size"],
                              config["mpi"]["local_size"],
                              per_instance_comm.Get_rank(),
                              per_instance_comm.Get_size(),
                              mpi_comm,
                              per_instance_comm,
                              instance,
                              num_instances)


class DistributedMeanAbsoluteError(object):
    def __init__(self):
        self.reset()

        self.mae_op = torch.nn.L1Loss(reduction="sum")

    def reset(self):
        self._items: int = 0
        self._error: float = 0.0

    def update(self, y: torch.Tensor, y_hat: torch.Tensor) -> float:
        self._error += self.mae_op(y, y_hat)
        self._items += y.numel()

    def get_value(self,
                  distributed: bool = False,
                  pg_handler: Optional[Any] = None) -> float:
        if self._items == 0:
            return 0

        if not distributed:
            return (self._error / self._items).item()
        else:
            input_tensor = torch.tensor([self._error, self._items],
                                        device=self._error.device)
            torch.distributed.all_reduce(input_tensor, group=pg_handler)
            return (input_tensor[0] / input_tensor[1]).item()


class ProfilerSection(object):
    def __init__(self, name: str, profile: bool = False):
        self.profile = profile
        self.name = name

    def __enter__(self):
        if self.profile:
            torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, *args, **kwargs):
        if self.profile:
            # torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()


class CudaExecutionTimer(object):
    def __init__(self, stream: Optional[torch.cuda.Stream] = None):
        self._stream = stream
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self._start_event.record(stream=self._stream)
        return self

    def __exit__(self, *args, **kwargs):
        self._end_event.record(stream=self._stream)

    def time_elapsed(self) -> float:
        self._end_event.synchronize()
        return self._start_event.elapsed_time(self._end_event)


class ExecutionTimer(object):
    def __init__(self, name: str, profile: bool = False):
        self._name = name
        self._profile = profile

    def __enter__(self):
        torch.cuda.nvtx.range_push(self._name)
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        torch.cuda.nvtx.range_pop()
        self._stop_time = time.time()

    def start(self):
        self._start_time = time.time()

    def time_elapsed(self) -> float:
        if not hasattr(self, "_stop_time"):
            return time.time() - self._start_time

        return self._stop_time - self._start_time


libcudart = None


def cudaProfilerStart():
    global libcudart
    libcudart = cdll.LoadLibrary('libcudart.so')
    libcudart.cudaProfilerStart()


def cudaProfilerStop():
    global libcudart
    assert libcudart, "libcudart undefined or None. cudaProfilerStart should be called before cudaProfilerStop"
    libcudart.cudaProfilerStop()


class Logger(object):
    def __init__(self,
                 distenv: DistributedEnv,
                 timestamp: str,
                 experiment_id: str):

        if int(os.getenv("THROUGPUT_RUN", "0")):
            _instance = int(os.getenv("EXPERIMENT_ID", "0"))
            logger_path = os.path.join("/results", f"{timestamp}_{_instance}.log")
            mllog.config(filename=logger_path)
        self.constants = mllog.constants
        self.distenv = distenv

        self.mllogger = MLLoggerWrapper(PyTCommunicationHandler(), value=None)

    def event(self, *args, **kwargs):
        self._print(self.mllogger.event, *args, **kwargs)

    def start(self, *args, **kwargs):
        self._print(self.mllogger.start, *args, **kwargs)

    def end(self, *args, **kwargs):
        self._print(self.mllogger.end, *args, **kwargs)

    def stop(self, *args, **kwargs):
        self.end(*args, **kwargs)

    def _print(self, logger, key, value=None,
               metadata=None, namespace=None, stack_offset=3, uniq=True):
        if metadata is not None:
            metadata["instance"] = self.distenv.instance
        else:
            metadata = {"instance": self.distenv.instance}

        logger(key=key, value=value, metadata=metadata,
               stack_offset=stack_offset, unique=uniq)
