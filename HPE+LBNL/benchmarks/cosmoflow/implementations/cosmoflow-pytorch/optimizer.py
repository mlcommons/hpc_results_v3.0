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

import utils

from torch.optim.optimizer import required
from omegaconf import DictConfig

import itertools
from typing import Callable, List, Optional, Tuple

from apex.optimizers import FusedSGD


def get_optimizer(train_config: DictConfig, model: torch.nn.Module) -> Tuple[object, object]:
    TRAIN_CONFIG_CLASS_MAPPER = {"sgd": torch.optim.SGD,
                                 "fixed_sgd": FixedSGDOptimizer,
                                 "fused_sgd": FusedSGD}
    kwargs = {}
    # if train_config["optimizer"] == "fused_sgd":
    #    kwargs = {"set_grad_none": True}
    if train_config["lr_sched"]["warmup_epochs"] > 0:
        initial_lr = train_config["lr_sched"]["init_lr"]
    else:
        initial_lr = train_config["lr_sched"]["lr"]

    if isinstance(model, nn.parallel.DistributedDataParallel):
        model_impl = model.module
    else:
        model_impl = model

    wd_parameters = [model_impl.dense1.get_parameter("weight"),
                     model_impl.dense2.get_parameter("weight")]
    non_wd_parameters = itertools.chain(model_impl.conv_seq.parameters(),
                                        model_impl.output.parameters(),
                                        [model_impl.dense1.get_parameter("bias"),
                                         model_impl.dense2.get_parameter("bias")])

    if train_config["weight_decay"] > 0.0:
        optimizer = TRAIN_CONFIG_CLASS_MAPPER[train_config["optimizer"]](
            [{"params": wd_parameters, "weight_decay": train_config["weight_decay"]},
             {"params": non_wd_parameters, "weight_decay": 0.0}],
            lr=initial_lr, momentum=train_config["momentum"], **kwargs)
    else:
        optimizer = TRAIN_CONFIG_CLASS_MAPPER[train_config["optimizer"]](
            lr=initial_lr,
            params=model.parameters(),
            momentum=train_config["momentum"],
            weight_decay=train_config["weight_decay"], **kwargs)
    utils.logger.event(key=utils.logger.constants.OPT_WEIGHT_DECAY,
                       value=train_config["weight_decay"])
    utils.logger.event(key=utils.logger.constants.OPT_NAME,
                       value=utils.logger.constants.SGD)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        LRSchedulerCalculator(init_lr=initial_lr,
                              target_lr=train_config["lr_sched"]["lr"],
                              warmup_epochs=train_config["lr_sched"]["warmup_epochs"],
                              decay_epochs=list(
                                  train_config["lr_sched"]["decay_steps"]),
                              decay_values=list(train_config["lr_sched"]["decay_values"])))
    return optimizer, lr_scheduler


class LRSchedulerCalculator(object):
    def __init__(self,
                 init_lr: float,
                 target_lr: float,
                 warmup_epochs: int,
                 decay_epochs: List[int],
                 decay_values: List[float]):
        self._init_lr = init_lr
        self._target_lr = target_lr
        self._warmup_epochs = warmup_epochs
        self._decay_epochs = decay_epochs
        self._decay_values = decay_values
        self._current_idx = 0
        self._current_decay = 1.0

        self._target_mult = self._target_lr / self._init_lr

        utils.logger.event(key=utils.logger.constants.OPT_BASE_LR,
                           value=self._target_lr)
        utils.logger.event(key=utils.logger.constants.OPT_LR_WARMUP_EPOCHS,
                           value=self._warmup_epochs)
        utils.logger.event(key=utils.logger.constants.OPT_LR_WARMUP_FACTOR,
                           value=int(self._target_lr / self._init_lr))
        utils.logger.event(key=utils.logger.constants.OPT_LR_DECAY_BOUNDARY_EPOCHS,
                           value=self._decay_epochs)
        utils.logger.event(key=utils.logger.constants.OPT_LR_DECAY_FACTOR,
                           value=self._decay_values)

    def disable(self) -> None:
        print("Disabling LRScheduler")
        self._target_mult = 0

    def __call__(self, epoch: int) -> float:
        if self._target_mult == 0:
            return 0

        if epoch < self._warmup_epochs and self._warmup_epochs > 0:
            return ((self._target_mult - 1) * epoch / self._warmup_epochs) + 1
        else:
            while (self._current_idx < len(self._decay_epochs) and
                   self._decay_epochs[self._current_idx] <= epoch):
                self._current_decay = self._decay_values[self._current_idx]
                self._current_idx += 1
            return self._current_decay * self._target_mult


class FixedSGDOptimizer(torch.optim.SGD):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(
                            torch.zeros_like(p.grad, requires_grad=False))
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            result = fixed_sgd(params_with_grad,
                               d_p_list,
                               momentum_buffer_list,
                               weight_decay=weight_decay,
                               momentum=momentum,
                               lr=lr)

            # update momentum_buffers in state
            for p, momentum_buffer in result:
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


@torch.jit.script
def single_sgd_update(param: torch.Tensor,
                      d_p: torch.Tensor,
                      mom: torch.Tensor,
                      *,
                      weight_decay: float,
                      momentum: float,
                      lr: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if weight_decay != 0:
        d_p = d_p.add(param, alpha=weight_decay)

    if momentum != 0:
        mom = mom.mul(momentum).add(d_p, alpha=lr)
        d_p = mom

    return d_p, mom


@torch.jit.script
def fixed_sgd(params: List[torch.Tensor],
              d_p_list: List[torch.Tensor],
              momentum_buffer_list: List[torch.Tensor],
              *,
              weight_decay: float,
              momentum: float,
              lr: float) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """
    result: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        buf = momentum_buffer_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = buf.mul(momentum).add(d_p, alpha=lr)
            d_p = buf
        param.sub_(d_p)
        result.append((param, buf))
    return result
