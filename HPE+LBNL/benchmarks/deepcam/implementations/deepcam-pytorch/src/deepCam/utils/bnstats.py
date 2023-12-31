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

import torch
import torch.nn as nn
import torch.distributed as dist

from utils import comm

class BatchNormStatsSynchronize(nn.Module):

    def __init__(self, model, reduction = "mean", inplace = False, normalize_before_add = False):
        super(BatchNormStatsSynchronize, self).__init__()
        
        # store args
        self.reduction = reduction
        self.inplace = inplace
        self.normalize_before_add = normalize_before_add
        
        # get world size
        self.comm_size = 1
        if dist.is_initialized():
            self.comm_size = comm.get_data_parallel_size()
        
        # create tensor lists
        self.list_means = []
        self.list_vars = []
        self.paramlist = []
        self.paramcount = 0
        for m in model.modules():
            if hasattr(m, "running_mean"):
                self.paramlist.append(m.running_mean)
                self.list_means.append(m.running_mean)
                self.paramcount += m.running_mean.numel()
            if hasattr(m, "running_var"):
                self.paramlist.append(m.running_var)
                self.list_vars.append(m.running_var)
                self.paramcount += m.running_var.numel()
        
        assert self.paramcount > 0
        
        self.device = self.paramlist[0].device
        
        # create big tensor which holds all stats
        self.buffer = torch.zeros((self.paramcount), 
                                   dtype = self.paramlist[0].dtype, 
                                   device = self.device,
                                   requires_grad = False)
        
        # create views to parameter buffers
        self.paramviews = []
        offset = 0
        for param in self.paramlist:
            numel = param.numel()
            self.paramviews.append(self.buffer[offset:offset+numel].view(numel))
            offset += numel
        
        # replace the original parameters with views to the new buffer
        if self.inplace:
            idx = 0
            self.list_means = []
            self.list_vars = [] 
            for m in model.modules():
                if hasattr(m, "running_mean"):
                    #m.running_mean = self.paramviews[idx]
                    del m.running_mean
                    m.register_buffer("running_mean", self.paramviews[idx], persistent=True)
                    self.paramviews[idx].fill_(0.)
                    self.list_means.append(self.paramviews[idx])
                    idx += 1
                if hasattr(m, "running_var"):
                    #m.running_var = self.paramviews[idx]
                    del m.running_var
                    m.register_buffer("running_var", self.paramviews[idx], persistent=True)
                    self.paramviews[idx].fill_(1.)
                    self.list_vars.append(self.paramviews[idx])
                    idx += 1


    def _copy_params_to_buffer(self):
        for idx, param in enumerate(self.paramlist):
            self.paramviews[idx].copy_(param)


    def _copy_params_from_buffer(self):
        for idx, param in enumerate(self.paramlist):
            param.copy_(self.paramviews[idx])

    def impute(self):
        pass
        #for p in self.list_means:
        #    #if torch.any(torch.isnan(p)):
        #    #    print(f"Rank {dist.get_rank()}: NaN in mean", p)
        #    torch.nan_to_num(p, nan=0., posinf=0., neginf=0., out=p)

        #for p in self.list_vars:
        #    #if torch.any(torch.isnan(p)):
        #    #    print(f"Rank {dist.get_rank()}: NaN in var", p)
        #    torch.nan_to_num(p, nan=1., posinf=1., neginf=1., out=p)

    def synchronize(self):
        if dist.is_initialized():
            # sync the device before
            torch.cuda.synchronize()
            
            with torch.no_grad():
                if not self.inplace:
                    self._copy_params_to_buffer()
        
                if self.reduction == "mean":
                    # normalize before?
                    if self.normalize_before_add:
                        self.buffer /= float(self.comm_size)

                    # sum
                    dist.all_reduce(self.buffer, op=dist.ReduceOp.SUM,
                                    group=comm.get_data_parallel_group())

                    # normalize after?
                    if not self.normalize_before_add:
                        self.buffer /= float(self.comm_size)
                else:
                    raise NotImplementedError(f"Error, reduction {self.reduction} not supported.")
            
                if not self.inplace:
                    self._copy_params_from_buffer()

            # sync the device after
            # torch.cuda.synchronize()

