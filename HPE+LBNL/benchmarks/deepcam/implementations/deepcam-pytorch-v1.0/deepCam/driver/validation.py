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
import gc

# numpy
import numpy as np

# torch
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.distributed as dist

# custom stuff
from utils import metric

# import wandb
try:
    import wandb
except ImportError:
    pass


class Validator(object):
    def __init__(self, pargs, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device
        
        # extract relevant parameters
        self.batch_size = pargs.local_batch_size_validation
        self.enable_jit = pargs.enable_jit
        self.enable_amp = (pargs.precision_mode == "amp")
        self.force_fp16 = (pargs.precision_mode == "fp16")
        self.enable_nhwc = pargs.enable_nhwc

        # disable for now because of pool sharing bug
        self.enable_graph = False #pargs.enable_graph
        self.enable_dali = (not pargs.data_format == "hdf5")
        
        # set that to None
        self.graph = None

        # check if we have groupbn somewhere in the model
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
        
        # set to eval
        self.model.eval() 
        
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
        with torch.no_grad():
            with amp.autocast(enabled = self.enable_amp):
                if self.jit_scriptable:
                    self.model = torch.jit.script(self.model)
                else:
                    self.model = torch.jit.trace(self.model, input_example, check_trace = False)
                
                self.criterion = torch.jit.script(self.criterion)
        
    
    def _warmup(self, input_shape, label_shape, warmup_stream = None, num_warmup = 20):
        
        # set to eval
        self.model.eval()
        
        # extract or create capture stream
        stream = torch.cuda.Stream() if warmup_stream is None else warmup_stream

        # create input:
        input_example = torch.zeros((self.batch_size, *input_shape), dtype=torch.float32, device=self.device)
        input_example.normal_()
        label_example = torch.zeros((self.batch_size, *label_shape), dtype=torch.int64, device=self.device)

        # convert to half if requested
        if self.force_fp16:
            input_example = input_example.half()
        
        # we need to convert to NHWC if necessary
        if self.enable_nhwc:
            input_example = input_example.contiguous(memory_format = torch.channels_last)
        
        # wait for ambient stream before starting capture
        stream.wait_stream(torch.cuda.current_stream())
        
        # no grads please
        with torch.no_grad():
        
            # enter stream context
            with torch.cuda.stream(stream):
                
                # warmup:
                for _ in range(num_warmup):
                    with amp.autocast(enabled = self.enable_amp):
                        output = self.model(input_example)
                        loss = self.criterion(output, label_example)
                        
        torch.cuda.current_stream().wait_stream(stream)
    
    
    def _capture(self, input_shape, label_shape, graph_stream = None, num_warmup = 20, graph_pool = None):

        # exit if we do not capture
        if not self.enable_graph:
            return
        
        # set to eval
        self.model.eval()
        
        # extract or create capture stream
        capture_stream = torch.cuda.Stream() if graph_stream is None else graph_stream
        
        # create input:
        self.static_input = torch.zeros((self.batch_size, *input_shape), dtype=torch.float32, device=self.device)
        self.static_input.normal_()
        self.static_label = torch.zeros((self.batch_size, *label_shape), dtype=torch.int64, device=self.device)

        # convert to half if requested
        if self.force_fp16:
            self.static_input = self.static_input.half()
        
        # we need to convert to NHWC if necessary
        if self.enable_nhwc:
            self.static_input = self.static_input.contiguous(memory_format = torch.channels_last)
            
        # wait for ambient stream before starting capture
        capture_stream.wait_stream(torch.cuda.current_stream())
        
        # no grads please
        with torch.no_grad():
        
            # enter stream context
            with torch.cuda.stream(capture_stream):

                # warmup:
                for _ in range(num_warmup):
                    with amp.autocast(enabled = self.enable_amp):
                        output = self.model(self.static_input)
                        loss = self.criterion(output, self.static_label)
                
                # sync streams
                capture_stream.synchronize()

                # clean up
                if num_warmup > 0:
                    del output,loss

                gc.collect()
                torch.cuda.empty_cache()
            
                # create graph
                self.graph = torch.cuda._Graph()
            
                # capture
                if graph_pool is not None:
                    self.graph.capture_begin(pool = graph_pool)
                else:
                    self.graph.capture_begin()
                with amp.autocast(enabled = self.enable_amp):
                    self.static_output = self.model(self.static_input)
                    self.static_loss = self.criterion(self.static_output, self.static_label)
                self.graph.capture_end()
        
        torch.cuda.current_stream().wait_stream(capture_stream)


    def preprocess(self, input_shape, label_shape, scaffolding_stream = None, graph_pool = None):        
        # compile
        self._compile(input_shape)
        
        # warmup
        self._warmup(input_shape, label_shape, warmup_stream = scaffolding_stream, num_warmup = 10)

        # capture
        self._capture(input_shape, label_shape, graph_stream = scaffolding_stream, num_warmup = 0, graph_pool = graph_pool)


    def eval(self):
        self.model.eval()


    def train(self):
        self.model.train()


    def evaluate(self, validation_loader, comm_rank):
                
        # set net to eval
        red_buffer = torch.zeros((3), dtype=torch.float32, device=self.device, requires_grad=False)
        count_sum_val = red_buffer[0].view(1)
        loss_sum_val = red_buffer[1].view(1)
        iou_sum_val = red_buffer[2].view(1)
        
        # set to eval
        self.model.eval()
        
        # no grad section:
        with torch.no_grad():
        
            # iterate over validation sample
            step_val = 0
            for inputs_val, label_val, filename_val in validation_loader:

                if not self.enable_dali:
                    #send to device
                    inputs_val = inputs_val.to(self.device)
                    label_val = label_val.to(self.device)

                if inputs_val.numel() == 0:
                    # we are done
                    continue

                # store samples in batch
                num_samples = inputs_val.shape[0]
                if num_samples < self.batch_size:
                    inputs_padding = (0,0, 0,0, 0,0, 0,self.batch_size-num_samples)
                    inputs_val = F.pad(inputs_val, inputs_padding, "constant", 0)
                    label_padding = (0,0, 0,0, 0,self.batch_size-num_samples)
                    label_val = F.pad(label_val, label_padding, "constant", 0)
                    
                # convert to half if requested
                if self.force_fp16:
                    inputs_val = inputs_val.half()
                    
                # to NHWC
                if self.enable_nhwc:
                    N, H, W, C = (self.batch_size, 768, 1152, 16)
                    inputs_val = torch.as_strided(inputs_val, size=[N, C, H, W], stride = [C*H*W, 1, W*C, C])
        
                if self.graph is None:
            
                    # forward pass
                    with amp.autocast(enabled = self.enable_amp):
                        outputs_val = self.model.forward(inputs_val)
                
                        # Compute loss
                        loss_val = self.criterion(outputs_val, label_val)

                else:
                    self.static_input.copy_(inputs_val)
                    self.static_label.copy_(label_val)
                    self.graph.replay()
            
                    # copy variables
                    loss_val = self.static_loss.clone()
                    outputs_val = self.static_output.clone()
                    
                # Compute score
                if self.enable_nhwc:
                    outputs_val = outputs_val.contiguous(memory_format = torch.contiguous_format)

                # strip tensor here if requested
                if num_samples < self.batch_size:
                    outputs_val = outputs_val[:num_samples, ...]
                    label_val = label_val[:num_samples, ...]
                    
                predictions_val = torch.argmax(torch.softmax(outputs_val.float(), 1), 1)
                iou_val = metric.compute_score_new(predictions_val, label_val, num_classes=3)
                iou_sum_val += iou_val * num_samples
                
                # accumulate loss
                loss_sum_val += loss_val * num_samples

                #increase counter
                count_sum_val += num_samples
        
        # average the validation loss
        if dist.is_initialized():
            dist.all_reduce(red_buffer, op=dist.ReduceOp.SUM, async_op=False)
            
        count_red = count_sum_val.item()
        loss_avg_val = loss_sum_val.item() / count_red
        iou_avg_val = iou_sum_val.item() / count_red
        
        return loss_avg_val, iou_avg_val


def validate(pargs, comm_rank, comm_size,
             step, epoch, validator,
             validation_loader, 
             logger, have_wandb):
    
    logger.log_start(key = "eval_start", metadata = {'epoch_num': epoch+1})

    # evaluate
    loss_avg_val, iou_avg_val = validator.evaluate(validation_loader, comm_rank)

    # print results
    logger.log_event(key = "eval_accuracy", value = iou_avg_val, metadata = {'epoch_num': epoch+1, 'step_num': step})
    logger.log_event(key = "eval_loss", value = loss_avg_val, metadata = {'epoch_num': epoch+1, 'step_num': step})

    # log in wandb
    if have_wandb and (comm_rank == 0):
        wandb.log({"eval_loss": loss_avg_val}, step=step)
        wandb.log({"eval_accuracy": iou_avg_val}, step=step)

    stop_training = False
    if (iou_avg_val >= pargs.target_iou):
        logger.log_event(key = "target_accuracy_reached", value = pargs.target_iou, metadata = {'epoch_num': epoch+1, 'step_num': step})
        stop_training = True

    logger.log_end(key = "eval_stop", metadata = {'epoch_num': epoch+1})
    
    return stop_training
