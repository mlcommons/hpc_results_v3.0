# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

# base stuff
import os
import sys
import gc

# numpy
import numpy as np

# torch
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel

# custom stuff
from utils import metric

# import wandb
try:
    import wandb
except ImportError:
    pass


class Trainer(object):
    def __init__(self, pargs, model, criterion, optimizer, grad_scaler, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.gscaler = grad_scaler
        self.scheduler = scheduler
        self.device = device
        self.enable_dali = (not pargs.data_format == "hdf5")

        # check for distributed lamb:
        have_distributed_lamb = True
        try:
            from apex.contrib.optimizers.distributed_fused_lamb import DistributedFusedLAMB
        except:
            have_distributed_lamb = False
            
        if have_distributed_lamb and isinstance(self.optimizer, DistributedFusedLAMB):
            self.enable_distributed_lamb = True
            self.optimizer.set_is_accumulation_step(False)
        else:
            self.enable_distributed_lamb = False
            
        # we need this for distlamb
        if self.enable_distributed_lamb:
            # we need that in order for it to work with async graph capture
            self.lr_cpu = torch.tensor([0.], dtype=torch.float32, device='cpu').pin_memory()
            
        # extract relevant parameters
        self.batch_size = pargs.local_batch_size
        self.enable_jit = pargs.enable_jit
        self.enable_amp = (pargs.precision_mode == "amp")
        self.force_fp16 = (pargs.precision_mode == "fp16")
        self.enable_nhwc = pargs.enable_nhwc
        self.enable_graph = pargs.enable_graph
        
        # set that to None
        self.graph = None

        # check if model is scriptable
        self.jit_scriptable = True
        for m in self.model.modules():
            if hasattr(m, "jit_scriptable"):
                self.jit_scriptable = self.jit_scriptable and m.jit_scriptable
                if not self.jit_scriptable:
                    break


    def _compile(self, input_shape):

        # exit if we do not compile
        if not self.enable_jit:
            return
        
        # set model to train just to be sure
        self.model.train()
        
        # input example
        input_example = torch.zeros((self.batch_size, *input_shape), dtype=torch.float32, device=self.device)
        input_example.normal_()

        # convert to half if requested
        if self.force_fp16:
            input_example = input_example.half()
        
        # we need to convert to NHWC if necessary
        if self.enable_nhwc:
           input_example  = input_example.contiguous(memory_format = torch.channels_last)
        
        # compile the model
        with amp.autocast(enabled = self.enable_amp):
            # extract the right thing to jit
            model_handle = self.model if not isinstance(self.model, DistributedDataParallel) else self.model.module
            
            # GBN is not scriptable, we need to workaround here
            if self.jit_scriptable:
                model_handle = torch.jit.script(model_handle)
            else:
                model_handle = torch.jit.trace(model_handle, input_example, check_trace = False)

            # the criterion is always scriptable
            self.criterion = torch.jit.script(self.criterion)
    
    
    def _warmup(self, input_shape, label_shape, warmup_stream = None, num_warmup = 20):

        # set model to train just to be sure
        self.model.train()
        
        # extract or create stream
        stream = torch.cuda.Stream() if warmup_stream is None else warmup_stream
        
        # create input:
        input_example = torch.zeros((self.batch_size, *input_shape), dtype=torch.float32, device=self.device)
        input_example.normal_()
        label_example  = torch.zeros((self.batch_size, *label_shape), dtype=torch.int64, device=self.device)

        # convert to half if requested
        if self.force_fp16:
            input_example = input_example.half()
        
        # we need to convert to NHWC if necessary
        if self.enable_nhwc:
           input_example  = input_example.contiguous(memory_format = torch.channels_last)
        
        # wait for ambient stream before starting capture
        stream.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(stream):
            
            # warmup:
            for _ in range(num_warmup):
                
                self.optimizer.zero_grad()
                
                with amp.autocast(enabled = self.enable_amp):
                    output = self.model(input_example)
                    loss = self.criterion(output, label_example)

                # distributed lamb init
                if self.enable_distributed_lamb:
                    self.optimizer._lazy_init_stage1()
                
                self.gscaler.scale(loss).backward()

                # distributed lamb finalize
                if self.enable_distributed_lamb:
                    self.optimizer._lazy_init_stage2()
                    self.optimizer.complete_reductions()
        
        torch.cuda.current_stream().wait_stream(stream)
        
        
    def _capture(self, input_shape, label_shape, graph_stream = None, num_warmup = 20, graph_pool = None):

        # exit if we do not capture
        if not self.enable_graph:
            return
        
        # set model to train just to be sure
        self.model.train()
        
        # extract or create capture stream
        capture_stream = torch.cuda.Stream() if graph_stream is None else graph_stream
        
        # create input:
        self.static_input = torch.zeros((self.batch_size, *input_shape), dtype=torch.float32, device=self.device)
        self.static_input.normal_()
        self.static_label  = torch.zeros((self.batch_size, *label_shape), dtype=torch.int64, device=self.device)

        # convert to half if requested
        if self.force_fp16:
            self.static_input = self.static_input.half()
        
        # we need to convert to NHWC if necessary
        if self.enable_nhwc:
           self.static_input  = self.static_input.contiguous(memory_format = torch.channels_last)
           
        # wait for ambient stream before starting capture
        capture_stream.wait_stream(torch.cuda.current_stream())
        
        # enter stream context
        with torch.cuda.stream(capture_stream):
            
            # warmup:
            for _ in range(num_warmup):
                self.optimizer.zero_grad()
                
                # FW pass
                with amp.autocast(enabled = self.enable_amp):
                    output = self.model(self.static_input)
                    loss = self.criterion(output, self.static_label)

                # distributed lamb work here
                if self.enable_distributed_lamb:
                    self.optimizer._lazy_init_stage1() 
                    
                # BW pass
                self.gscaler.scale(loss).backward()
                
                # distributed lamb postprocessing
                if self.enable_distributed_lamb:
                    self.optimizer._lazy_init_stage2()
                    self.optimizer.set_global_scale(self.gscaler._get_scale_async())
                    self.optimizer.complete_reductions()
                    self.gscaler.step(self.optimizer)
                    self.gscaler.update()
            
            # sync streams
            capture_stream.synchronize()

            # clean up
            if num_warmup > 0:
                del output,loss
            gc.collect()
            torch.cuda.empty_cache()   
            
            # create graph
            self.graph = torch.cuda._Graph()

            # zero grads before capture:
            self.model.zero_grad(set_to_none=True)
            
            # start capture
            if graph_pool is not None:
                self.graph.capture_begin(pool = graph_pool)
            else:
                self.graph.capture_begin()
            
            # preprocessing
            #self.optimizer.zero_grad() # not necessary according to Michael
            #self.static_scale = self.gscaler._scale
            
            # FW pass
            with amp.autocast(enabled = self.enable_amp):
                self.static_output = self.model(self.static_input)
                self.static_loss = self.criterion(self.static_output, self.static_label)
            
            # BW pass
            self.gscaler.scale(self.static_loss).backward()

            # should also be done
            # distributed lamb postprocessing
            if self.enable_distributed_lamb:
                self.optimizer.set_global_scale(self.gscaler._get_scale_async())
                self.optimizer.complete_reductions()
                self.gscaler.step(self.optimizer)
                self.gscaler.update()
                            
            # end capture
            self.graph.capture_end()
        
        torch.cuda.current_stream().wait_stream(capture_stream)


    def preprocess(self, input_shape, label_shape, scaffolding_stream = None, graph_pool = None):
        
        # compile
        self._compile(input_shape)

        # warmup
        self._warmup(input_shape, label_shape, warmup_stream = scaffolding_stream, num_warmup = 10)

        # capture
        self._capture(input_shape, label_shape, graph_stream = scaffolding_stream, num_warmup = 0, graph_pool = graph_pool)
        

    def step(self, inputs, label):

        # set model to train to be sure
        self.model.train()
        
        # convert input if requested
        if self.force_fp16:
            inputs = inputs.half()
        
        # to NHWC
        if self.enable_nhwc:
            N, H, W, C = (self.batch_size, 768, 1152, 16)
            inputs = torch.as_strided(inputs, size=[N, C, H, W], stride=[C*H*W, 1, W*C, C])
    
        if self.graph is None:
    
            with amp.autocast(enabled = self.enable_amp):
                outputs = self.model.forward(inputs)
                loss = self.criterion(outputs, label)
            
            # prepare optimizer
            self.optimizer.zero_grad()

            # backward pass
            self.gscaler.scale(loss).backward()

            # postprocess
            if self.enable_distributed_lamb:
                self.optimizer.set_global_scale(self.gscaler._get_scale_async())
                self.optimizer.complete_reductions()

            # update scaler
            self.gscaler.step(self.optimizer)
            self.gscaler.update()
            
        else:            
            # run graph
            self.static_input.copy_(inputs)
            self.static_label.copy_(label)
            #self.static_scale.copy_(self.gscaler._scale)
            self.graph.replay()

            # DEBUG
            ## postprocess
            #if self.enable_distributed_lamb:
            #    self.optimizer.complete_reductions()
            #    self.optimizer.set_global_scale(self.gscaler._get_scale_async())
            # DEBUG

            if not self.enable_distributed_lamb:
                self.gscaler.step(self.optimizer)
                self.gscaler.update()
            
            # copy variables
            loss = self.static_loss.clone()
            outputs = self.static_output.clone()
        
        # get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # scheduler step if requested:
        if self.scheduler is not None:
            self.scheduler.step()
            if self.enable_distributed_lamb:
                self.lr_cpu[0] = current_lr
                self.optimizer._lr.copy_(self.lr_cpu[0])
            
        return loss, outputs, current_lr


def train_step(pargs, comm_rank, comm_size,
               step, epoch, trainer,
               train_loader,
               logger, have_wandb):
    
    # epoch loop
    for inputs, label, filename in train_loader:
    
        if not trainer.enable_dali:
            # send to device
            inputs = inputs.to(trainer.device)
            label = label.to(trainer.device)
            
        loss, outputs, current_lr = trainer.step(inputs, label)
    
        # step counter
        step += 1
    
        #log if requested
        if (step % pargs.logging_frequency == 0):
    
            # allreduce for loss
            loss_avg = loss.detach()
            if dist.is_initialized():
                dist.reduce(loss_avg, dst=0, op=dist.ReduceOp.SUM)
            loss_avg_train = loss_avg.item() / float(comm_size)
    
            # Compute score
            outputs = outputs.detach()
            if pargs.enable_nhwc:
                outputs = outputs.contiguous(memory_format = torch.contiguous_format)
            predictions = torch.argmax(torch.softmax(outputs, 1), 1)
            iou = metric.compute_score_new(predictions, label, num_classes=3)
            iou_avg = iou.detach()
            if dist.is_initialized():
                dist.reduce(iou_avg, dst=0, op=dist.ReduceOp.SUM)
            iou_avg_train = iou_avg.item() / float(comm_size)

            # log values
            logger.log_event(key = "learning_rate", value = current_lr, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.log_event(key = "train_accuracy", value = iou_avg_train, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.log_event(key = "train_loss", value = loss_avg_train, metadata = {'epoch_num': epoch+1, 'step_num': step})
    
            if have_wandb and (comm_rank == 0):
                wandb.log({"train_loss": loss_avg_train}, step = step)
                wandb.log({"train_accuracy": iou_avg_train}, step = step)
                wandb.log({"learning_rate": current_lr}, step = step)
            
    return step



def train_step_profile(pargs, comm_rank, comm_size,
                       step, epoch, trainer,
                       train_loader,
                       start_profiler, stop_profiler):

    # enable profiling
    with torch.autograd.profiler.emit_nvtx(enabled = True):
        
        # epoch loop
        train_iter = iter(train_loader)
        epoch_done = False
        while(True):

            if step == pargs.capture_range_start:
                start_profiler() 

            # step region
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"step_{step}")

            # IO region
            torch.cuda.nvtx.range_push(f"data_loading") 
            try:
                inputs, label, filename = next(train_iter)
            except StopIteration:
                epoch_done = True
            torch.cuda.nvtx.range_pop()
            if epoch_done:
                break
            
            if pargs.data_format == "hdf5":
                # send to device
                inputs = inputs.to(trainer.device)
                label = label.to(trainer.device)

            if not pargs.io_only:
                loss, outputs, current_lr = trainer.step(inputs, label)
            
            # step counter
            step += 1

            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

            if step >= pargs.capture_range_stop:
                stop_profiler()
                break

    return step
