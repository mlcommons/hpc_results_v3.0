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

import abc

import torch

from omegaconf import OmegaConf
from typing import Any

import utils


class Application(abc.ABC):
    def __init__(self, config: OmegaConf):
        self._config = config

    def setup(self) -> None:
        pass

    @abc.abstractmethod
    def run(self) -> Any:
        pass

    def exec(self) -> Any:
        self.setup()
        return self.run()


class MPIApplication(Application):
    def setup(self) -> None:
        if "mpi" in self._config:
            from socket import gethostname
            self._distenv = utils.DistributedEnv.create_from_mpi(self._config)
            self._master_host = self._distenv.instance_mpi_comm.allgather(
                gethostname())[0]
        else:
            self._distenv = utils.DistributedEnv.create_single()

        utils.setup_logger(self._distenv,
                           self._config["log"]["timestamp"],
                           self._config["log"]["experiment_id"])


class PytorchApplication(MPIApplication):
    def setup(self) -> None:
        super().setup()

    def init_ddp(self):
        if "mpi" in self._config:
            torch.distributed.init_process_group("NCCL", init_method=f"tcp://{self._master_host}:12345",
                                                 rank=self._distenv.rank,
                                                 world_size=self._distenv.size)
            torch.cuda.set_device(self._distenv.local_rank)
