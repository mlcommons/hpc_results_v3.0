# The MIT License (MIT)
#
# Copyright (c) 2020-2023 NVIDIA CORPORATION. All rights reserved.
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
import tempfile
import time

# numpy
import numpy as np

# torch
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel

# custom stuff
from utils import metric
from utils import comm

# custom logging
from utils.profile_logger import PLogger

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
        self.data_parallel_size = comm.get_data_parallel_size()

        # some debug modes
        self.ddp_mode = pargs.ddp_mode

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

        # check for MP lamb
        have_mp_lamb = True
        try:
            from apex.optimizers.fused_mixed_precision_lamb import FusedMixedPrecisionLamb
        except:
            have_mp_lamb = False
            
        if have_mp_lamb and isinstance(self.optimizer, FusedMixedPrecisionLamb):
            self.enable_mp_lamb = True
        else:
            self.enable_mp_lamb = False

        self.enable_gpu_scheduler = (self.enable_distributed_lamb or self.enable_mp_lamb)
            
        # we need this for distlamb
        if self.enable_gpu_scheduler:
            # we need that in order for it to work with async graph capture
            self.lr_cpu = torch.tensor([0.], dtype=torch.float32, device='cpu').pin_memory()

        if self.enable_mp_lamb and (self.scheduler is not None):
            # we need to fix what the scheduler screwed up: initializing the
            # former removed the lr from the gpu:
            for group, lr_gpu in zip(self.optimizer.param_groups, self.scheduler.group_lrs_gpu):
                group["lr"] = lr_gpu
            # we need to update the backups as well
            self.scheduler.group_lrs_backup = [x.clone() for x in self.scheduler.group_lrs_gpu]
        
        # extract relevant parameters
        self.batch_size = pargs.local_batch_size
        self.enable_jit = pargs.enable_jit
        self.enable_amp = True if "amp" in pargs.precision_mode else False
        self.amp_dtype = torch.bfloat16 if pargs.precision_mode == "amp-bf16" else torch.float16
        self.force_fp16 = (pargs.precision_mode == "fp16")
        self.enable_nhwc = pargs.enable_nhwc
        self.enable_graph = pargs.enable_graph

        # save some states
        self.gscaler_state = self.gscaler.state_dict()
        
        # set that to None
        self.graph = None
                
    def _save_state(self, filename):
        torch.save({"model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scaler_state": self.gscaler.state_dict()}, filename)
        return

        
    def _load_state(self, filename, include_exp_avg=False):
        checkpoint = torch.load(filename)
        # loading model is straightforward
        self.model.load_state_dict(checkpoint["model_state"])
        self.gscaler.load_state_dict(checkpoint["scaler_state"])
        # in case of optimizer we have to be more careful.
        # load_state_dict replaces existing per-param state tensors with incoming tensors,
        # rather than copy_ing from the incoming tensors.
        # This means load_state_dict is not safe to use on an optimizer that's already been graph-captured.
        # print("self.optimizer.state_dict()", self.optimizer.state_dict())
        # print("saved state", checkpoint["optimizer_state"])
        for group, cgroup in zip(self.optimizer.param_groups, checkpoint["optimizer_state"]["param_groups"]):
            for key in ["step", "lr"]:
                group[key].copy_(cgroup[key])
                
            if include_exp_avg:
                # Quick and dirty version of what load_state_dict does
                # (less robust: assumes all params and states in old and new state dict match perfectly)
                for new_state, old_state in zip(group["params"], cgroup["params"]):
                    #new_state = self.optimizer.state[p]
                    #old_state = checkpoint["optimizer_state"]["state"][i]
                    new_state["exp_avg"].copy_(old_state["exp_avg"])
                    new_state["exp_avg_sq"].copy_(old_state["exp_avg_sq"])

        return

    def reset_state(self):

        # sync
        torch.cuda.synchronize()
        
        # do everything in no-grad region
        with torch.no_grad():
            
            # get model handle
            if dist.is_initialized():
                model_handle = self.model.module
            else:
                model_handle = self.model

            # reset weights
            model_handle.init_weight()

            # reset batchnorms
            for m in model_handle.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
                else:
                    if hasattr(m, "running_mean"):
                        m.running_mean.zero_()
                    if hasattr(m, "running_var"):
                        m.running_var.fill_(1)
                    if hasattr(m, "num_batches_tracked"):
                        m.num_batches_tracked.zero_()
        
            # reset scheduler
            self.scheduler.reset()

            # optimizer stats
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    # gradient
                    if param.grad is not None:
                        param.grad.zero_()

                    # state
                    state = self.optimizer.state[param]
                    if "exp_avg" in state:
                        state["exp_avg"].zero_()
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"].zero_()

            # gradient scaler
            self.gscaler.load_state_dict(self.gscaler_state)

        # sync again
        torch.cuda.synchronize()

        return
    

    def _compile(self, input_shape):

        # the criterion is always jittable
        self.criterion = torch.jit.script(self.criterion)
        
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
        #with amp.autocast(enabled = self.enable_amp, dtype=self.amp_dtype):
            # extract the right thing to jit
            #self.model = torch.compile(self.model, options={"triton.cudagraphs": False})
            #self.criterion = torch.compile(self.criterion, options={"triton.cudagraphs": False})
            
            #if isinstance(self.model, DistributedDataParallel):            
            #    # GBN is not scriptable, we need to workaround here
            #    #if self.jit_scriptable:
            #    #    self.model.module = torch.jit.script(self.model.module)
            #    #else:
            #    #    self.model.module = torch.jit.trace(self.model.module, input_example, check_trace = False)
            #    #self.model.module = torch.compile(self.model.module, mode="max-autotune")
            #    self.model = torch.compile(self.model, mode="max-autotune")
            #else:
            #    # GBN is not scriptable, we need to workaround here
            #    #if self.jit_scriptable:
            #    #    self.model = torch.jit.script(self.model)
            #    #else:
            #    #    self.model = torch.jit.trace(self.model, input_example, check_trace = False)
            #    self.model = torch.compile(self.model, mode="max-autotune")
            #
            # the criterion is always scriptable
            #self.criterion = torch.jit.script(self.criterion)
    
    
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
                
                with amp.autocast(enabled = self.enable_amp, dtype=self.amp_dtype):
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
        
        return
    
        
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
                with amp.autocast(enabled = self.enable_amp, dtype=self.amp_dtype):
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
                    lr = self.scheduler.group_lrs_gpu[0]
                    self.scheduler.step_gpu()
                    self.optimizer._lr.copy_(self.scheduler.group_lrs_gpu[0])

                # mp lamb postprocessing
                if self.enable_mp_lamb:
                    self.gscaler.step(self.optimizer)
                    self.gscaler.update()
                    lr = self.scheduler.group_lrs_gpu[0]
                    self.scheduler.step_gpu()
                    for idg,group in enumerate(self.optimizer.param_groups):
                        group["lr"].copy_(self.scheduler.group_lrs_gpu[idg])
                        
            # sync streams
            capture_stream.synchronize()
            
            # clean up
            if num_warmup > 0:
                del output,loss
                if self.enable_gpu_scheduler:
                    del lr
            gc.collect()
            torch.cuda.empty_cache()   
            
            # create graph
            self.graph = torch.cuda.CUDAGraph()

            # zero grads before capture:
            self.model.zero_grad(set_to_none=True)
            
            # start capture
            if graph_pool is not None:
                self.graph.capture_begin(pool = graph_pool)
            else:
                self.graph.capture_begin()
                        
            # FW pass
            with amp.autocast(enabled = self.enable_amp, dtype=self.amp_dtype):
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
                self.static_lr = self.scheduler.group_lrs_gpu[0]
                self.scheduler.step_gpu()
                self.optimizer._lr.copy_(self.scheduler.group_lrs_gpu[0])

            # mp lamb postprocessing
            if self.enable_mp_lamb:
                self.gscaler.step(self.optimizer)
                self.gscaler.update()
                self.static_lr = self.scheduler.group_lrs_gpu[0]
                self.scheduler.step_gpu()
                for idg,group in enumerate(self.optimizer.param_groups):
                    group["lr"].copy_(self.scheduler.group_lrs_gpu[idg])

            # end capture
            self.graph.capture_end()

            # sync up
            capture_stream.synchronize()

        
        torch.cuda.current_stream().wait_stream(capture_stream)
        
        return


    def preprocess(self, input_shape, label_shape, scaffolding_stream = None, graph_pool = None):
        
        # compile
        self._compile(input_shape)

        # warmup
        self._warmup(input_shape, label_shape, warmup_stream = scaffolding_stream, num_warmup = 10)

        # capture
        self._capture(input_shape, label_shape, graph_stream = scaffolding_stream, num_warmup = 1, graph_pool = graph_pool)

        return
        

    def step(self, inputs, label, disable_scheduler=False):
        
        # set model to train to be sure
        self.model.train()
        
        # convert input if requested
        if self.force_fp16:
            inputs = inputs.half()
        
        # to NHWC
        if self.enable_nhwc:
            N, H, W, C = (self.batch_size, 768, 1152, 16)
            inputs = torch.as_strided(inputs, size=[N, C, H, W], stride=[C*H*W, 1, W*C, C])

        # perform training step
        if self.graph is None:
    
            with amp.autocast(enabled = self.enable_amp, dtype=self.amp_dtype):
                outputs = self.model.forward(inputs)
                loss = self.criterion(outputs, label)
            
            # prepare optimizer
            self.model.zero_grad(set_to_none=True)

            # backward pass
            self.gscaler.scale(loss).backward()
            
            # postprocess
            if self.enable_distributed_lamb:
                self.optimizer.set_global_scale(self.gscaler._get_scale_async())
                self.optimizer.complete_reductions()

            # check for ddp mode
            if self.ddp_mode == "sync":
                dist.barrier(device_ids=[comm.get_local_rank()], group=comm.get_data_parallel_group())
                
            # update scaler
            self.gscaler.step(self.optimizer)
            self.gscaler.update()

            if self.enable_gpu_scheduler:
                self.static_lr = self.scheduler.group_lrs_gpu[0]
            
        else:            
            # run graph
            self.static_input.copy_(inputs)
            self.static_label.copy_(label)
            self.graph.replay()

            # check for ddp mode
            if self.ddp_mode == "sync":
                dist.barrier(device_ids=[comm.get_local_rank()], group=comm.get_data_parallel_group())

            if not self.enable_gpu_scheduler:
                self.gscaler.step(self.optimizer)
                self.gscaler.update()
    
            # copy variables
            loss = self.static_loss.detach().clone()
            outputs = self.static_output.detach().clone()
        
        # get current learning rate
        if self.enable_gpu_scheduler:
            current_lr = self.static_lr.detach().clone()
        else:
            current_lr = self.optimizer.param_groups[0]['lr']
            
        # scheduler step if requested:
        if (self.scheduler is not None) and (not disable_scheduler):

            # switch depending on where scheduler runs
            if not self.enable_gpu_scheduler:
                self.scheduler.step()
            else:
                if self.graph is None:
                    self.scheduler.step_gpu()
                    if self.enable_distributed_lamb:
                        self.optimizer._lr.copy_(self.scheduler.group_lrs_gpu[0])
                    else:
                        for idg,group in enumerate(self.optimizer.param_groups):
                            group["lr"].copy_(self.scheduler.group_lrs_gpu[idg])
            
        return loss, outputs, current_lr


def train_epoch(pargs, comm_rank, comm_size,
                step, epoch, trainer,
                train_loader,
                logger, have_wandb,
                disable_scheduler=False):

    # get logger
    plog = PLogger.getInstance()
    
    # create a buffer for cpu information in pinned memory
    loss_avg_train_cpu = torch.zeros((1), dtype=torch.float32, device=torch.device("cpu"), requires_grad=False).pin_memory()
    iou_avg_train_cpu = torch.zeros((1), dtype=torch.float32, device=torch.device("cpu"), requires_grad=False).pin_memory()
    current_lr_cpu = torch.zeros((1), dtype=torch.float32, device=torch.device("cpu"), requires_grad=False).pin_memory() 
    
    # epoch loop
    steps_in_epoch = 0
    start_time = time.perf_counter_ns()
    for inputs, label, filename in train_loader:

        # log step
        plog.event(plog.INTERVAL_START, key="step_start", metadata={"step_num": step, "epoch_num": epoch})
        plog.nvml_log_start()
        
        if not trainer.enable_dali:
            # send to device
            inputs = inputs.to(trainer.device)
            label = label.to(trainer.device)
            
        loss, outputs, current_lr = trainer.step(inputs, label, disable_scheduler=disable_scheduler)
    
        # step counter
        step += 1
        steps_in_epoch += 1


        #log if requested
        if (pargs.logging_frequency > 0) and (step % pargs.logging_frequency == 0):

            # log the analysis
            plog.event(plog.INTERVAL_START, key="summarize_start", metadata={"step_num": step, "epoch_num": epoch})
            
            # wait for the device to finish
            if trainer.enable_distributed_lamb or trainer.enable_mp_lamb:
                torch.cuda.synchronize()
            
            # allreduce for loss
            loss_avg = loss.detach().clone()
            if dist.is_initialized():
                dist.all_reduce(loss_avg, op=dist.ReduceOp.AVG, group=comm.get_data_parallel_group())
            loss_avg_train_cpu.copy_(loss_avg, non_blocking=True)
    
            # Compute score
            outputs_avg = outputs.detach().clone()
            if pargs.enable_nhwc:
                outputs_avg = outputs_avg.contiguous(memory_format = torch.contiguous_format)
            predictions = torch.argmax(torch.softmax(outputs_avg, 1), 1)
            iou = metric.compute_score_new(predictions, label, num_classes=3)
            iou_avg = iou.detach()
            if dist.is_initialized():
                dist.all_reduce(iou_avg, op=dist.ReduceOp.AVG, group=comm.get_data_parallel_group())
            iou_avg_train_cpu.copy_(iou_avg, non_blocking=True)

            # check for lr val:
            if isinstance(current_lr, torch.Tensor):
                current_lr_cpu.copy_(current_lr, non_blocking=True)
            else:
                current_lr_cpu.copy_(torch.tensor(current_lr))

            # make sure we copied everything
            torch.cuda.current_stream().synchronize()
                
            # log values
            logger.event(key = "learning_rate", value = current_lr_cpu.item(), metadata = {'epoch_num': epoch+1, 'step_num': step})
            #logger.event(key = "learning_rate_gpu", value = current_lr_gpu, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.event(key = "train_accuracy", value = iou_avg_train_cpu.item(), metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.event(key = "train_loss", value = loss_avg_train_cpu.item(), metadata = {'epoch_num': epoch+1, 'step_num': step})
    
            if have_wandb and (comm_rank == comm.get_data_parallel_root()):
                wandb.log({"train_loss": loss_avg_train_cpu.item()}, step = step)
                wandb.log({"train_accuracy": iou_avg_train_cpu.item()}, step = step)
                wandb.log({"learning_rate": current_lr_cpu.item()}, step = step)

            # log the analysis
            plog.event(plog.INTERVAL_END, key="summarize_stop", metadata={"step_num": step, "epoch_num": epoch})

        # log step
        plog.nvml_log_stop()
        plog.event(plog.INTERVAL_END, key="step_stop", metadata={"step_num": step, "epoch_num": epoch})

    # also profile cleanup phase
    plog.event(plog.INTERVAL_START, key="epoch_summary_start", metadata={"step_num": step, "epoch_num": epoch})
        
    # end of epoch logging
    # wait for the device to finish
    if trainer.enable_distributed_lamb or trainer.enable_mp_lamb:
        torch.cuda.synchronize()
    
    # allreduce for loss
    loss_avg = loss.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(loss_avg, op=dist.ReduceOp.AVG, group=comm.get_data_parallel_group())
    loss_avg_train_cpu.copy_(loss_avg, non_blocking=True) 

    # Compute score
    outputs_avg = outputs.detach().clone()
    if pargs.enable_nhwc:
        outputs_avg = outputs_avg.contiguous(memory_format = torch.contiguous_format)
    predictions = torch.argmax(torch.softmax(outputs_avg, 1), 1)
    iou = metric.compute_score_new(predictions, label, num_classes=3)
    iou_avg = iou.detach()
    if dist.is_initialized():
        dist.all_reduce(iou_avg, op=dist.ReduceOp.AVG, group=comm.get_data_parallel_group())
    iou_avg_train_cpu.copy_(iou_avg, non_blocking=True)

    # check for lr val:
    if isinstance(current_lr, torch.Tensor):
        current_lr_cpu.copy_(current_lr, non_blocking=True)
    else:
        current_lr_cpu.copy_(torch.tensor(current_lr))

    # make sure we copied everything
    torch.cuda.current_stream().synchronize()

    # end time
    end_time = time.perf_counter_ns()

    # compute throughput:
    throughput = (steps_in_epoch * trainer.batch_size * trainer.data_parallel_size) / ((end_time - start_time) * 10**(-9))
    
    # finalize logging
    plog.event(plog.INTERVAL_END, key="epoch_summary_stop", metadata={"step_num": step, "epoch_num": epoch})

    # log values
    logger.event(key = "learning_rate", value = current_lr_cpu.item(), metadata = {'epoch_num': epoch+1, 'step_num': step})
    logger.event(key = "train_accuracy", value = iou_avg_train_cpu.item(), metadata = {'epoch_num': epoch+1, 'step_num': step})
    logger.event(key = "train_loss", value = loss_avg_train_cpu.item(), metadata = {'epoch_num': epoch+1, 'step_num': step})
    logger.event(key = 'tracked_stats', value = {"throughput": throughput}, metadata = {'epoch': epoch+1, 'step': step})
    
    if have_wandb and (comm_rank == comm.get_data_parallel_root()):
        wandb.log({"train_loss": loss_avg_train_cpu.item()}, step = step)
        wandb.log({"train_accuracy": iou_avg_train_cpu.item()}, step = step)
        wandb.log({"learning_rate": current_lr_cpu.item()}, step = step)
            
    return step



def train_epoch_profile(pargs, comm_rank, comm_size,
                        step, epoch, trainer,
                        train_loader,
                        disable_scheduler,
                        start_profiler, stop_profiler,
                        record_shapes=False):

    # enable profiling
    with torch.autograd.profiler.emit_nvtx(enabled = True,
                                           record_shapes=record_shapes):
        
        # epoch loop
        train_iter = iter(train_loader)
        epoch_done = False
        while(True):
            
            if step == pargs.capture_range_start:
                if (comm_rank == comm.get_data_parallel_root()):
                    print("Starting Profiler")
                if start_profiler is not None:
                    start_profiler() 

            # start step region
            torch.cuda.nvtx.range_push(f"step_{step}")

            # start IO region
            torch.cuda.nvtx.range_push(f"data_loading") 
            try:
                inputs, label, filename = next(train_iter)
            except StopIteration:
                epoch_done = True
            # end IO region
            torch.cuda.nvtx.range_pop()
            if epoch_done:
                break
            
            if pargs.data_format == "hdf5":
                # send to device
                inputs = inputs.to(trainer.device)
                label = label.to(trainer.device)

            if not pargs.io_only:
                loss, outputs, current_lr = trainer.step(inputs, label, disable_scheduler=disable_scheduler)
            
            # step counter
            step += 1

            # end step region
            torch.cuda.nvtx.range_pop()

            if step >= pargs.capture_range_stop:
                if (comm_rank == comm.get_data_parallel_root()):
                    print("Stopping Profiler")
                if stop_profiler is not None:
                    stop_profiler()
                break

    return step
