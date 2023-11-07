import torch
import torch.distributed as dist

from .cuda_graph import capture_graph


def capture_model(pargs, net, input_shape, device, graph_stream=None, mode="model"):
    
    if mode == "module":
        # capture model by modules
    
        # xception layers
        train_example = [torch.ones( (pargs.local_batch_size, *input_shape), dtype=torch.float32, device=device)]

        # NHWC
        if pargs.enable_nhwc:
            train_example = [x.contiguous(memory_format = torch.channels_last) for x in train_example]
        
        net.xception_features = capture_graph(net.xception_features, 
                                              tuple(t.clone() for t in train_example), 
                                              graph_stream = graph_stream,
                                              warmup_iters = 10,
                                              use_amp = pargs.enable_amp and not pargs.enable_jit)
                                              
        ## bottleneck
        #train_example = [torch.ones( (pargs.local_batch_size, 2048, 48, 72), 
        #                              dtype=torch.float32,
        #                              device=device)]
        #
        #if pargs.enable_nhwc:
        #    train_example = [x.contiguous(memory_format = torch.channels_last) for x in train_example]
        #
        #net.bottleneck = capture_graph(net.bottleneck,
        #                               tuple(t.clone().requires_grad_() for t in train_example),
        #                               graph_stream = graph_stream,
        #                               warmup_iters = 10,
        #                               use_amp = pargs.enable_amp and not pargs.enable_jit)
        
        ## upsample
        #train_example = [torch.ones( (pargs.local_batch_size, 256,  48,  72), dtype=torch.float32, device=device),
        #                 torch.ones( (pargs.local_batch_size,  48, 192, 288), dtype=torch.float32, device=device)]
        #
        #if pargs.enable_nhwc:
        #    train_example = [x.contiguous(memory_format = torch.channels_last) for x in train_example]
        #
        #net.upsample = capture_graph(net.upsample, 
        #                             tuple(t.clone().requires_grad_() for t in train_example), 
        #                             graph_stream = graph_stream,
        #                             warmup_iters = 10,
        #                             use_amp = pargs.enable_amp and not pargs.enable_jit)
                                        
    elif mode == "model":
        
        #print("Graphing start")
        
        ## stream setup
        #stream = torch.cuda.Stream() if graph_stream is None else graph_stream
        #ambient_stream = torch.cuda.current_stream()
        #stream.wait_stream(ambient_stream)
        #
        ## get functional args
        #train_example = tuple(x.clone() for x in [torch.ones( (pargs.local_batch_size, *input_shape), dtype=torch.float32, device=device)])
        #module_params = tuple(net.parameters())
        #functional_args = train_example + module_params
        #
        #with torch.cuda.stream(stream):
        
            ## NHWC
            #if pargs.enable_nhwc:
            #    train_example = [x.contiguous(memory_format = torch.channels_last) for x in train_example]
            #
            #warmup_iters = 10
            #for _ in range(warmup_iters):
            #    #with torch.cuda.amp.autocast(enabled = pargs.enable_amp):
            #    outputs  = net.module(*train_example)
            #    
            #    outputs_was_tensor = isinstance(outputs, torch.Tensor)
            #    outputs = (outputs,) if outputs_was_tensor else outputs
            #
            #    outputs_require_grad = tuple(o for o in outputs if o.requires_grad)
            #    args_require_grad = tuple(i for i in functional_args if i.requires_grad)
            #    buffer_incoming_grads = tuple(torch.empty_like(o) if o.requires_grad else None for o in outputs)
            #    needed_incoming_grads = tuple(b for b in buffer_incoming_grads if b is not None)
            #    #torch.cuda.nvtx.range_push("autograd.grad")
            #    grad_inputs = torch.autograd.grad(outputs_require_grad,
            #                                      args_require_grad,
            #                                      needed_incoming_grads,
            #                                      only_inputs=True,
            #                                      allow_unused=False)
            #                                      
            #if warmup_iters > 0:
            #    del outputs, outputs_require_grad, args_require_grad, buffer_incoming_grads, needed_incoming_grads, grad_inputs
            #
            #
            ## FW pass
            #fwd_graph = torch.cuda._Graph()
            #fwd_graph.capture_begin()
            ##with torch.cuda.amp.autocast(enabled = pargs.enable_amp):
            #outputs  = net.module(*train_example)
            #fwd_graph.capture_end()
        
            # some bookeeping
            #outputs = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
            #outputs_require_grad = tuple(o for o in outputs if o.requires_grad)
            #args_require_grad = tuple(i for i in functional_args if i.requires_grad)
            #buffer_incoming_grads = tuple(torch.empty_like(o) if o.requires_grad else None for o in outputs)
            #needed_incoming_grads = tuple(b for b in buffer_incoming_grads if b is not None)
            #
            ## BW pass
            #bwd_graph = torch.cuda._Graph()
            #bwd_graph.capture_begin(pool=fwd_graph.pool())
            #grad_inputs = torch.autograd.grad(outputs_require_grad,
            #                                  args_require_grad,
            #                                  needed_incoming_grads,
            #                                  only_inputs=True,
            #                                  allow_unused=False)
            #bwd_graph.capture_end()
            
            ## bookeeping
            #buffer_inputs = tuple(i.detach() for i in functional_args)
            #buffer_outputs = tuple(o.detach().requires_grad_(o.requires_grad) for o in outputs)
            #
            ## Constructs a list suitable for returning from Graphed.backward:
            ## Inserts Nones in gradient slots for inputs that don't expect a grad.
            #buffer_grad_inputs = []
            #grad_idx = 0
            #for arg in functional_args:
            #    if arg.requires_grad:
            #        buffer_grad_inputs.append(grad_inputs[grad_idx])
            #    grad_idx += 1
            #    else:
            #        buffer_grad_inputs.append(None)
            #buffer_grad_inputs = tuple(buffer_grad_inputs)
        
        #ambient_stream.wait_stream(stream)
        #print("Graphing done")
        
        train_example = [torch.ones( (pargs.local_batch_size, *input_shape), dtype=torch.float32, device=device)]
        
        # NHWC
        if pargs.enable_nhwc:
            train_example = [x.contiguous(memory_format = torch.channels_last) for x in train_example]
        
        net = capture_graph(net, 
                            tuple(t.clone() for t in train_example), 
                            graph_stream = graph_stream,
                            warmup_iters = 10,
                            use_amp = pargs.enable_amp and not pargs.enable_jit)
    
    return net
    
    
