# Copyright (c) 2021-2023 NVIDIA CORPORATION. All rights reserved.
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

from omegaconf.dictconfig import DictConfig
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP


from typing import Iterator, Tuple, Optional
from model.utils import Convolution3DLayout

import utils


DataIter = Iterator[Tuple[torch.Tensor, torch.Tensor]]

TRAINING_DATASET_ITEMS = 524288


def _should_mark_profiling(epoch: int, step: int, config: str, start: bool) -> bool:
    temp = config.split(',', 1)
    temp = [temp[0]] + temp[1].split('-', 1)

    return int(temp[0]) == epoch and int(temp[2 - int(start)]) == step


def _convert_format(input_tensor: torch.Tensor) -> torch.Tensor:
    return input_tensor.log1p()


class Trainer(object):
    def __init__(self,
                 config: DictConfig,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                 distenv: utils.DistributedEnv,
                 amp: bool = False,
                 enable_profiling: bool = False):
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.score_fn = utils.DistributedMeanAbsoluteError()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self._config = config
        self._enable_profiling = enable_profiling
        self._distenv = distenv
        self._model_layout = Convolution3DLayout(config["model"]["layout"])
        self._data_layout = Convolution3DLayout(config["data"]["data_layout"])

        self.zeroing_stream = torch.cuda.Stream()
        self.prefetch_stream = torch.cuda.Stream()
        self.last_scale = None

        self._amp = amp
        if self._amp:
            self.scaler_ = torch.cuda.amp.GradScaler()

    def train_step(self,
                   x: torch.Tensor,
                   y_hat: torch.Tensor) -> None:
        with utils.ProfilerSection("training step", self._enable_profiling):
            if hasattr(self.model, "graph_capture"):
                with utils.ProfilerSection("copy", self._enable_profiling):
                    with torch.cuda.stream(self.zeroing_stream):
                        self.model.zero_capture.replay()

                    self.model.static_input_data.copy_(x)
                    self.model.static_input_label.copy_(y_hat)
                    torch.cuda.current_stream().wait_stream(self.zeroing_stream)

                with utils.ProfilerSection("replay", self._enable_profiling):
                    self.model.graph_capture.replay()
                self.scaler_.step(self.optimizer)
                self.scaler_.update()
            else:
                self.optimizer.zero_grad()
                if self._amp:
                    with torch.cuda.amp.autocast():
                        y = self.model(x)
                        loss = self.loss_fn(y, y_hat)
                    self.scaler_.scale(loss).backward()
                    self.scaler_.step(self.optimizer)
                    self.scaler_.update()
                else:
                    y = self.model(x)
                    loss = self.loss_fn(y, y_hat)
                    loss.backward()
                    self.optimizer.step()

    def train_epoch(self,
                    train_iter: DataIter,
                    epoch: int):
        with utils.ProfilerSection(f"training epoch #{epoch}",
                                   self._enable_profiling):
            self.model.train()
            should_run = True
            current_step = 0

            try:
                with torch.cuda.stream(self.prefetch_stream):
                    input_data = next(train_iter)
            except StopIteration:
                should_run = False

            while should_run:
                if ("profile_range" in self._config and
                        _should_mark_profiling(epoch, current_step, self._config["profile_range"], start=True)):
                    utils.cudaProfilerStart()
                with utils.ProfilerSection("convert", self._enable_profiling):
                    torch.cuda.current_stream().wait_stream(self.prefetch_stream)
                    data = self._convert(input_data[0])
                    data = _convert_format(data)
                    label = input_data[1]

                try:
                    with torch.cuda.stream(self.prefetch_stream):
                        input_data = next(train_iter)
                except StopIteration:
                    should_run = False

                self.train_step(data, label)

                if ("profile_range" in self._config and
                        _should_mark_profiling(epoch, current_step, self._config["profile_range"], start=False)):
                    utils.cudaProfilerStop()
                current_step += 1
            self.lr_scheduler.step()

    def eval_epoch(self,
                   eval_iter: DataIter,
                   epoch: int) -> float:
        with utils.ProfilerSection(f"eval epoch #{epoch}",
                                   self._enable_profiling):
            self.model.eval()
            self.score_fn.reset()

            with torch.no_grad():
                for step, input_data in enumerate(eval_iter):
                    data = self._convert(input_data[0])
                    label = input_data[1]
                    data = _convert_format(data)
                    if self._amp:
                        with torch.cuda.amp.autocast():
                            y = self.model(data)
                    else:
                        y = self.model(data)

                    self.score_fn.update(y.float(), label)

        return self.score_fn.get_value(distributed=not self._distenv.is_single,
                                       pg_handler=None)

    def epoch_step(self,
                   train_iter: DataIter,
                   eval_iter: DataIter,
                   epoch: int,
                   eval_only: bool = False) -> float:
        utils.logger.start(key=utils.logger.constants.EPOCH_START,
                           metadata={'epoch_num': epoch + 1,
                                     "lr": self.lr_scheduler.get_last_lr()})
        with utils.CudaExecutionTimer() as train_latency:
            if not eval_only:
                self.train_epoch(train_iter, epoch)

        utils.logger.start(key=utils.logger.constants.EVAL_START,
                           metadata={'epoch_num': epoch + 1})
        with utils.CudaExecutionTimer() as eval_latency:
            validation_score = self.eval_epoch(eval_iter, epoch)
        utils.logger.start(key=utils.logger.constants.EVAL_STOP,
                           metadata={'epoch_num': epoch + 1})

        train_epoch_elapsed = train_latency.time_elapsed()
        eval_epoch_elapsed = eval_latency.time_elapsed()
        utils.logger.stop(key=utils.logger.constants.EPOCH_STOP,
                          metadata={'epoch_num': epoch + 1,
                                    'training_epoch_latency': train_epoch_elapsed,
                                    'eval_epoch_latency': eval_epoch_elapsed})
        utils.logger.event(key='tracked_stats',
                           value={"throughput": TRAINING_DATASET_ITEMS /
                                  (train_epoch_elapsed / 1000)},
                           metadata={'epoch_num': epoch+1,
                                     'step': epoch})
        utils.logger.event(key='eval_error',
                               value=validation_score,
                               metadata={'epoch_num': epoch + 1})
        # torch.cuda.synchronize()
        return validation_score

    def warmup(self, capture_stream: Optional[torch.cuda.Stream] = None):
        if self._amp:
            begin_scale = self.scaler_.get_scale()

        backup_weights = {}

        # Warmup kernels
        # for _ in range(10):
        #    input_data = torch.rand((self._config["data"]["batch_size"],
        #                             *list(self._config["data"]["sample_shape"])),
        #                            dtype=torch.float32,
        #                            device=next(self.model.parameters()).device).transpose(1, -1)
        #    input_label = torch.rand((self._config["data"]["batch_size"],
        #                              *list(self._config["data"]["target_shape"])),
        #                             dtype=torch.float32, device=input_data.device)
        #    if self._model_layout.channel_last:
        #        input_data = input_data.to(
        #            memory_format=torch.channels_last_3d)
        #
        #    if self._amp:
        #        with torch.cuda.amp.autocast():
        #            y = self.model(input_data)
        #            loss = self.loss_fn(y, input_label)
        #        self.scaler_.scale(loss).backward()
        #    else:
        #        y = self.model(input_data)
        #        loss = self.loss_fn(y, input_label)
        #        loss.backward()
        #
        #    self.optimizer.zero_grad(set_to_none=True)
        # torch.cuda.synchronize()

        if capture_stream is not None:
            # Warmup cuda graphs

            static_input_data = torch.rand((self._config["data"]["batch_size"],
                                            *list(self._config["data"]["sample_shape"])),
                                           dtype=torch.float32,
                                           device=next(self.model.parameters()).device).transpose(1, -1)
            if self._model_layout.channel_last:
                static_input_data = static_input_data.to(
                    memory_format=torch.channels_last_3d)

            static_input_label = torch.rand((self._config["data"]["batch_size"],
                                            *list(self._config["data"]["target_shape"])),
                                            dtype=torch.float32, device=static_input_data.device)

            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for param in self.model.parameters():
                    backup_weights[param] = param.clone()

                for _ in range(50):
                    self.optimizer.zero_grad()

                    if self._amp:
                        with torch.cuda.amp.autocast():
                            y = self.model(static_input_data)
                            loss = self.loss_fn(y, static_input_label)
                        self.scaler_.scale(loss).backward()
                    else:
                        y = self.model(static_input_data)
                        loss = self.loss_fn(y, static_input_label)
                        loss.backward()

            torch.cuda.current_stream().wait_stream(capture_stream)

            if self._config["model"]["cuda_graph"] == True:
                self.optimizer.zero_grad()
                self.model.graph_capture = torch.cuda.CUDAGraph()
                self.model.zero_capture = torch.cuda.CUDAGraph()
                self.model.static_input_data = static_input_data
                self.model.static_input_label = static_input_label

                with torch.cuda.graph(self.model.zero_capture):
                    self.optimizer.zero_grad()

                with torch.cuda.graph(self.model.graph_capture):
                    if self._amp:
                        with torch.cuda.amp.autocast():
                            y = self.model(static_input_data)
                            loss = self.loss_fn(y, static_input_label)
                        self.scaler_.scale(loss).backward()
                    else:
                        y = self.model(static_input_data)
                        loss = self.loss_fn(y, static_input_label)
                        loss.backward()

        if self._amp:
            self.scaler_.update(begin_scale)

        for _ in range(10):
            self.optimizer.step()
            torch.distributed.all_reduce(static_input_data)

        with torch.no_grad():
            for param in self.model.parameters():
                param.copy_(backup_weights[param])

    def _convert(self, tensor: torch.Tensor) -> torch.Tensor:
        strides = tensor.stride()
        shape = tensor.shape

        return torch.as_strided(tensor,
                                (shape[0], shape[-1], *shape[1:-1]),
                                (strides[0], strides[-1], *strides[1:-1]))
