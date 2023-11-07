# The MIT License (MIT)
#
# Modifications Copyright (c) 2020-2023 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

import math
from collections import Counter

import torch
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepLRWarmup(_LRScheduler):

    @torch.jit.ignore
    def __init__(self, optimizer, warmup_steps, warmup_factor,
                 milestones, gamma=0.1, last_epoch=-1,
                 device=torch.device(f"cuda:{torch.cuda.current_device()}"), verbose=False):
        
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.warmup_slope = 1./float(self.warmup_steps) if self.warmup_steps > 0 else 1.
        self.device = device
        self.gamma = gamma
        self.milestones_gpu = torch.tensor(milestones, dtype=torch.int64, device=self.device)
        self.milestones = Counter([x + self.warmup_steps + 1 for x in milestones])
        super(MultiStepLRWarmup, self).__init__(optimizer, last_epoch, verbose)
        self.last_epoch_gpu = torch.tensor(self.last_epoch, dtype=torch.int64, device=self.device)
        self.opt_params_on_gpu = isinstance(self.optimizer.param_groups[0]["lr"], torch.Tensor)
        # make sure the param groups have a step tensor:
        for group in self.optimizer.param_groups:
            if not "step" in group:
                if self.opt_params_on_gpu:
                    group["step"] = torch.Tensor([0], dtype=torch.int, device=self.device)
                else:
                    group["step"] = 0

        # copy on gpu just in case
        if self.opt_params_on_gpu:
            self.group_lrs_gpu = [group["lr"].detach().clone().to(self.device) for group in self.optimizer.param_groups]
            self.group_steps_gpu = [group["step"].detach().clone().to(self.device) for group in self.optimizer.param_groups]
        else:
            self.group_lrs_gpu = [group["lr"].detach().clone() if isinstance(group["lr"], torch.Tensor)
                                  else torch.tensor(group["lr"], dtype=torch.float32, device=self.device)
                                  for group in self.optimizer.param_groups]
            self.group_steps_gpu = [group["step"].detach().clone() if isinstance(group["step"], torch.Tensor)
                                    else torch.tensor(group["step"], dtype=torch.int, device=self.device) for group in self.optimizer.param_groups]

        # we need that for scheduler resets
        self.last_epoch_backup = self.last_epoch
        self.group_lrs_gpu_backup = [x.clone() for x in self.group_lrs_gpu]
        self.group_steps_gpu_backup = [x.clone() for x in self.group_steps_gpu]

        
    def reset(self):
        self.last_epoch = self.last_epoch_backup
        self.last_epoch_gpu.copy_(torch.tensor(self.last_epoch_backup, dtype=torch.int64))
        for group, lr_backup, step_backup in zip(self.optimizer.param_groups, self.group_lrs_gpu_backup, self.group_steps_gpu):
            if self.opt_params_on_gpu:
                group["lr"].copy_(lr_backup)
                group["step"].copy_(step_backup)
            else:
                group["lr"] = lr_backup.item()
                group["step"] = step_backup.item()
        self.group_lrs_gpu = [x.copy_(y) for x,y in zip(self.group_lrs_gpu, self.group_lrs_gpu_backup)]
        self.group_steps_gpu = [x.copy_(y) for x,y in zip(self.group_steps_gpu, self.group_steps_gpu_backup)]

        # we have to reset some other things too:
        for group in self.optimizer.param_groups:
            for param in group['params']:
                param.grad.zero_()
                state = self.optimizer.state[param]
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'].zero_()
        
        return

    
    #@torch.compile(options={"triton.cudagraphs": False})
    @torch.jit.export
    def get_lr_gpu(self):
        last_epoch_eff = self.last_epoch_gpu - self.warmup_steps
        cond_warm = (last_epoch_eff < 0).to(dtype=torch.float32, device=self.device)
        cond_decays = torch.where(last_epoch_eff > self.milestones_gpu, self.gamma, 1.)

        # compute individual lr
        # compute warmup lr
        if self.warmup_factor == 1.0:
            lr_warm = [base_lr * (self.last_epoch_gpu.to(dtype=torch.float32) * self.warmup_slope) for base_lr in self.base_lrs]
        else:
            lr_warm = [base_lr * ((self.warmup_factor - 1.) * self.last_epoch_gpu.to(dtype=torch.float32) * self.warmup_slope + 1.) for base_lr in self.base_lrs]

        # decay lr:
        fact = torch.prod(cond_decays)
        lr_decay = [base_lr * fact for base_lr in self.base_lrs]

        return [cond_warm * lw + (1. - cond_warm) * ld for lw,ld in zip(lr_warm, lr_decay)]

    #@torch.compile(options={"triton.cudagraphs": False})
    @torch.jit.export
    def step_gpu(self):
        # Instead of optimizer.param_groups['lr'],
        # update optimizer._lr to avoid sync
        self.last_epoch_gpu += 1
        for lhs, rhs in zip(self.group_lrs_gpu, self.get_lr_gpu()):
            lhs.copy_(rhs)
        return
        
    #@torch.compile(options={"triton.cudagraphs": False})
    @torch.jit.export
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)  
        
        # compute LR
        if self.last_epoch >= self.warmup_steps:
            # decay phase
            if self.last_epoch not in self.milestones:
                return [group['lr'] for group in self.optimizer.param_groups]
            return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                    for group in self.optimizer.param_groups] 

        else:
            # linear warmup phase
            if self.warmup_factor == 1.0:
                return [base_lr * (float(self.last_epoch) * self.warmup_slope)
                        for base_lr in self.base_lrs]
            else:
                return [base_lr * ((self.warmup_factor - 1.) * float(self.last_epoch) * self.warmup_slope + 1.)
                        for base_lr in self.base_lrs]



class CosineAnnealingLRWarmup(_LRScheduler):

    @torch.jit.ignore
    def __init__(self, optimizer, warmup_steps, warmup_factor,
                 T_max, eta_min=0, last_epoch=-1,
                 device=torch.cuda.current_device(), verbose=False):
        
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.warmup_slope = 1./float(self.warmup_steps) if self.warmup_steps > 0 else 1.
        self.T_max = T_max
        self.eta_min = eta_min
        self.device = device
        super(CosineAnnealingLRWarmup, self).__init__(optimizer, last_epoch, verbose)
        self.last_epoch_gpu = torch.tensor(self.last_epoch, dtype=torch.int64, device=self.device)
        self.opt_params_on_gpu = isinstance(self.optimizer.param_groups[0]["lr"], torch.Tensor)
        # make sure the param groups have a step tensor:
        for group in self.optimizer.param_groups:
            if not "step" in group:
                if self.opt_params_on_gpu:
                    group["step"] = torch.Tensor([0], dtype=torch.int, device=self.device)
                else:
                    group["step"] = 0 

        # copy on gpu just in case
        if self.opt_params_on_gpu:
            self.group_lrs_gpu = [group["lr"].detach().clone().to(self.device) for group in self.optimizer.param_groups]
            self.group_steps_gpu = [group["step"].detach().clone().to(self.device) for group in self.optimizer.param_groups]
        else:
            self.group_lrs_gpu = [group["lr"].detach().clone() if isinstance(group["lr"], torch.Tensor)
                                  else torch.tensor(group["lr"], dtype=torch.float32, device=self.device) for group in self.optimizer.param_groups]
            self.group_steps_gpu = [group["step"].detach().clone() if isinstance(group["step"], torch.Tensor)
                                    else torch.tensor(group["step"], dtype=torch.int, device=self.device) for group in self.optimizer.param_groups]
            

        # backup the lr params
        self.last_epoch_backup = self.last_epoch
        self.group_lrs_gpu_backup = [x.clone() for x in self.group_lrs_gpu]
        self.group_steps_gpu_backup = [x.clone() for x in self.group_steps_gpu]

    def reset(self):
        self.last_epoch = self.last_epoch_backup
        self.last_epoch_gpu.copy_(torch.tensor(self.last_epoch_backup, dtype=torch.int64))   
        
        for group, lr_backup, step_backup in zip(self.optimizer.param_groups, self.group_lrs_gpu_backup, self.group_steps_gpu):
            if self.opt_params_on_gpu:
                group["lr"].copy_(lr_backup)
                group["step"].copy_(step_backup)
            else:
                group["lr"] = lr_backup.item()
                group["step"] = step_backup.item()
        self.group_lrs_gpu = [x.copy_(y) for x,y in zip(self.group_lrs_gpu, self.group_lrs_gpu_backup)]
        self.group_steps_gpu = [x.copy_(y) for x,y in zip(self.group_steps_gpu, self.group_steps_gpu_backup)]

        # we have to reset some other things too:
        for group in self.optimizer.param_groups:
            for param in group['params']:
                param.grad.zero_()
                state = self.optimizer.state[param]
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'].zero_()
                            
        return
    
    #@torch.compile(options={"triton.cudagraphs": False})
    @torch.jit.export
    def get_lr_gpu(self):
        last_epoch_eff = self.last_epoch_gpu - self.warmup_steps
        cond_warm = (last_epoch_eff < 0).to(dtype=torch.float32, device=self.device)
        cond_phase = ((last_epoch_eff - 1 - self.T_max) % (2 * self.T_max) == 0).to(dtype=torch.float32, device=self.device)
        # compute warmup lr
        if self.warmup_factor == 1.0:
            lr_warm = [base_lr * (self.last_epoch_gpu.to(dtype=torch.float32) * self.warmup_slope) for base_lr in self.base_lrs]
        else:
            lr_warm = [base_lr * ((self.warmup_factor - 1.) * self.last_epoch_gpu.to(dtype=torch.float32) * self.warmup_slope + 1.) for base_lr in self.base_lrs]
        # compute cosine lr
        lr_cos1 = [group_lr + 0.5 * (base_lr - self.eta_min) * (1. - math.cos(math.pi / self.T_max)) for base_lr,group_lr in zip(self.base_lrs, self.group_lrs_gpu)]
        lr_cos2 = [(1. + torch.cos(math.pi * last_epoch_eff / self.T_max)) / (1. + torch.cos(math.pi * (last_epoch_eff - 1) / self.T_max)) * (group_lr - self.eta_min) + self.eta_min for group_lr in self.group_lrs_gpu]

        return [cond_warm * lw + (1. - cond_warm) * (cond_phase * lc1 + (1. - cond_phase) * lc2) for lw,lc1,lc2 in zip(lr_warm, lr_cos1, lr_cos2)]

    
    #@torch.compile(options={"triton.cudagraphs": False})
    @torch.jit.export
    def step_gpu(self):
        # Instead of optimizer.param_groups['lr'],
        # update optimizer._lr to avoid sync
        self.last_epoch_gpu += 1
        for lhs, rhs in zip(self.group_lrs_gpu, self.get_lr_gpu()):
            lhs.copy_(rhs)
        return
        
    #@torch.compile(options={"triton.cudagraphs": False})
    @torch.jit.export
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        # compute LR
        if self.last_epoch >= self.warmup_steps:
            # cosine phase
            last_epoch_eff = self.last_epoch - self.warmup_steps
            if last_epoch_eff == 0:
                return [group['lr'] for group in self.optimizer.param_groups]
            elif (last_epoch_eff - 1 - self.T_max) % (2 * self.T_max) == 0:
                return [group['lr'] + (base_lr - self.eta_min) *
                        (1 - math.cos(math.pi / self.T_max)) / 2
                        for base_lr, group in
                        zip(self.base_lrs, self.optimizer.param_groups)]
            return [(1 + math.cos(math.pi * last_epoch_eff / self.T_max)) /
                    (1 + math.cos(math.pi * (last_epoch_eff - 1) / self.T_max)) *
                    (group['lr'] - self.eta_min) + self.eta_min
                    for group in self.optimizer.param_groups]

        else:
            # linear warmup phase
            if self.warmup_factor == 1.0:
                return [base_lr * (float(self.last_epoch) * self.warmup_slope)
                        for base_lr in self.base_lrs]
            else:
                return [base_lr * ((self.warmup_factor - 1.) * float(self.last_epoch) * self.warmup_slope + 1.)
                        for base_lr in self.base_lrs]
