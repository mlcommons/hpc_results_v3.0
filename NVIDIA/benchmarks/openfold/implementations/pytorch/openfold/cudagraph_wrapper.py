# Copyright 2023 NVIDIA CORPORATION
#
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

from functools import lru_cache
from dataclasses import dataclass

import torch
import torch.nn as nn

class ParameterWrapper:
    def __init__(self):
        """Parameter Wrapper module. PyTorch `make_graphed_callables` and torch.cuda.CUDAGraph()
           API only support Tensor type as the input/output parameter. The ParameterWrapper module will
           help us to support arbitrary input/output type of cudagraph.

           You can see CudaGraphFunctionWrapper to see how to ParameterWrapper. 
        """

        self.args_root = self.Node()
        self.args_root.val = None
        self.args_root.type = None

        self.kwargs_root = self.Node()
        self.kwargs_root.val = None
        self.kwargs_root.type = None

        self.cur_root = None

        self.tensors = []
        self.literature = []
        self.nnModule = []
        
        self.tensor_idx = 0
        self.number_idx = 0
        self.module_idx = 0

        self.is_record = False
        self.check_guard = True

        
    @dataclass
    class Node:
        type: str = ""
        val = None
        name = None

    def _print(self, node, space=""):
        if node.type == "literature":
            print(space, "leaf_node = ", node.val, " name = ", node.name)
        elif node.type == "tensor":
            print(space, "leaf_node = tensor[", node.val.shape, "] name = ", node.name)
        elif isinstance(node.val, list):
            print(space, "range_node: [],  name = ", node.name)
            for val in node.val:
                self._print(val, space + 2 * " ")
        elif node.val == None:
            pass
        else:
            assert 0, "unknow error when print CudaGraphParameter"

    def print(self):
        print("==================")
        self._print(self.args_root)
        print("------------------")
        self._print(self.kwargs_root)
        print("==================")
        
    def get_next_node(self, root):
        if root.type in ["tensor", "literature", "module"]:
            yield root
        elif isinstance(root.val, list):
            yield root
            for i in range(len(root.val)):
                yield from self.get_next_node(root.val[i])
        else:
            yield None

    def _pack(self, node, tensors, numbers, modules):
        if node.type == "tensor":
            self.tensor_idx = self.tensor_idx + 1
            assert len(tensors) >= self.tensor_idx, "error in " + str(self.tensor_idx) + " want to access " + str(len(tensors))
            return (tensors[self.tensor_idx - 1], node.name)
        elif node.type == "literature":
            self.number_idx = self.number_idx + 1
            return (numbers[self.number_idx - 1], node.name)
        elif node.type == "module":
            self.module_idx = self.module_idx + 1
            return (modules[self.module_idx - 1], node.name)
        elif node.type == "dict":
            rval = {}
            for i in range(len(node.val)):
                next_node = next(self.next_node_generator)
                (v, k) = self._pack(next_node, tensors, numbers, modules)
                rval[k] = v
            return (rval, node.name)
        elif node.type == "list":
            rval = []
            for i in range(len(node.val)):
                next_node = next(self.next_node_generator)
                v, _ = self._pack(next_node, tensors, numbers, modules)
                rval.append(v)
            return (rval, node.name)
        elif node.type == "tuple":
            rval = ()
            for i in range(len(node.val)):
                next_node = next(self.next_node_generator)
                v, _ = self._pack(next_node, tensors, numbers, modules)
                rval = rval + (v, )
            return (rval, node.name)
        elif node.type == "set":
            rval = {}
            for i in range(node.range):
                next_node = next(self.next_node_generator)
                v, _ = self._pack(next_node, tensors, numbers, modules)
                rval.add(v)
            return (rval, node.name)
        else:
            assert 0, "unsupport node type " + str(node.type)

    def pack(self, tensors, numbers, modules=None):
        # given the flatten tensors and flatten literatures, the pack function 
        # will return the original datatype by the record history. 
        self.clear_idx()

        args = []
        kwargs = {}

        self.next_node_generator = self.get_next_node(self.args_root)
        node = next(self.next_node_generator)
        if node != None:
            args, _ = self._pack(node, tensors, numbers, modules)

        self.next_node_generator = self.get_next_node(self.kwargs_root)
        node = next(self.next_node_generator)
        if node != None:
            kwargs, _ = self._pack(node, tensors, numbers, modules)

        self.clear_idx()
        return (args, kwargs)

    def record_or_build_leaf(self, idx, type, name, val, is_root=False):
        # build the leaf in parameter tree. 
        if is_root:
            leaf_node = self.cur_root
        elif self.is_record:
            leaf_node = self.father_node.val[idx]
        else:
            leaf_node = self.Node()
            self.father_node.val.append(leaf_node)

        if self.is_record and self.check_guard:
            assert leaf_node.type == type, "please debug"
            assert leaf_node.name == name, "please debug"
        elif self.is_record:
            leaf_node.val =  val        
        else:
            leaf_node.type = type
            leaf_node.name = name
            leaf_node.val = val
        return leaf_node

    def record_or_build_range(self, idx, type, name, is_root=False):
        if is_root:
            range_node = self.cur_root
        elif self.is_record:
            range_node = self.father_node.val[idx]
        else:
            range_node = self.Node()
            self.father_node.val.append(range_node)

        if self.is_record and self.check_guard:
            assert range_node.type == type, "please debug"
            assert range_node.name == name, "please debug"
        elif not self.is_record:
            range_node.type = type
            range_node.name = name
            range_node.val =  []

        return range_node

    def clear_idx(self):
        self.tensor_idx = 0
        self.number_idx = 0
        self.module_idx = 0

    def clear_tree(self):
        self.args_root = self.Node()
        self.args_root.val = None
        self.args_root.type = None

        self.kwargs_root = self.Node()
        self.kwargs_root.val = None
        self.kwargs_root.type = None

        self.cur_root = None

    def _unpack(self, inp, idx, name, is_root=False):
        if inp == None:
            return
        elif isinstance(inp, torch.Tensor):
            # if the input is tensor, pick it to the self.tensors
            if self.is_record:
                self.tensors[self.tensor_idx] = inp
                self.tensor_idx = self.tensor_idx + 1
            else:   
                self.tensors.append(inp)
            self.record_or_build_leaf(idx, "tensor", name, inp, is_root)
        elif type(inp) in [float, int, bool, str]:
            if self.is_record:
                self.literature[self.number_idx] = inp
                self.number_idx = self.number_idx + 1
            else:   
                self.literature.append(inp)
            self.record_or_build_leaf(idx, "literature", name, inp, is_root)
        elif isinstance(inp, nn.Module):
            if self.is_record:
                self.nnModule[self.module_idx] = inp
                self.module_idx = self.module_idx + 1
            else: 
                self.nnModule.append(inp)
            self.record_or_build_leaf(idx, "module", name, inp, is_root)
        elif isinstance(inp, dict):
            keys, vals = inp.keys(), inp.values()
            node = self.record_or_build_range(idx, "dict", name, is_root)
            for i, (k, v) in enumerate(zip(keys, vals)):
                self.father_node = node
                self._unpack(v, i, name=k)
        elif isinstance(inp, tuple) or isinstance(inp, set) or isinstance(inp, list):
            node = self.record_or_build_range(idx, str(type(inp))[8:-2], name, is_root)
            for i, v in enumerate(inp):
                self.father_node = node
                self._unpack(v, i, name=None)
        else:
            assert 0, "unsupport data type " + str(type(inp))

    def unpack(self, args, kwarg):
        # given the arbirary parameters, the unpack function 
        # will record these parameters in this module and pick 
        # the tensor type in self.tensors.
        self.clear_idx()

        if not self.is_record:
            # if not the record mode, the parameter 
            # tree need to recapture.
            self.clear_tree()

        self.cur_root = self.args_root
        self.father_node = None
        self._unpack(args, idx=None, name=None, is_root=True)

        self.cur_root = self.kwargs_root
        self.father_node = None
        self._unpack(kwarg, idx=None, name=None, is_root=True)

        self.clear_idx()

class CudaGraphFunctionWrapper:
    """CudaGraph Function Wrapper module.

    Args:
        callable: Inner function.
        params_modules: the model parameters used in the callable function
        warmup_iterations: must be larger than 11 for ddp with cudagraph
        grad_guard: recapture if grad has changed
        shape_guard: recapture if shape has changed
        literature_guard: recapture if python variable (int, dict...) has changed
    """

    def __init__(
        self, 
        captured_object, 
        params_modules=None,
        warmup_iterations: int = 11,
        grad_guard: bool = True,
        shape_guard: bool = True,
        literature_guard: bool = False,
    ) -> None:
        super(CudaGraphFunctionWrapper, self).__init__()
        self.params_modules = params_modules
        self.warmup_times = warmup_iterations
        self.literature_guard = literature_guard
        self.grad_guard = grad_guard
        self.shape_guard = shape_guard


        # init runtime info and in/out/graph pool.
        self.func_runtimes = []
        self.graph_pool = []
        self.static_args_pool = []
        self.static_outputs_pool = []
        self.cached_idx = 0
        self.out_wrapper_pool = []


        # store the currect guards
        self.shapes = None
        self.grads = None
        self.literature = None

        self.captured_object = captured_object

        if not isinstance(self.captured_object, nn.Module):
            self.inner_fn = self.CudaGraphWrapperModule(self, self.captured_object, self.params_modules)

    def init_cudagraph(self):
        # set inner function (handle this seperately due to `deepcopy`)
        if isinstance(self.captured_object, nn.Module):
            self.inner_fn = self.CudaGraphWrapperModule(self, self.captured_object, [self.captured_object])

    class CudaGraphWrapperModule(nn.Module):
        # `make_graphed_callables` API doesn't capture the bw graph if the callable is not a nn.Module.
        # So, we wrap the captured_object (function) into a nn.Module. 
        def __init__(
            self,
            context,
            module,
            params_modules: list
        ) -> None:
            super(context.CudaGraphWrapperModule, self).__init__()
            self.inner_fn = module
            self.params_modules = params_modules
            self.context = context

        def parameters(self):
            # pytorch `make_graphed_callables` API get autograd args by call parameters(). 
            params = []
            for m in self.params_modules:
                params += list(m.parameters())
            return params

        def forward(self, *args):
            # args and output are torch.Tensor. So this forward can be captured by cudagraph.
            # notice: cudagraph only capture the device operator, so the `pack` and `unpack`
            # will not bring the cpu overhead after graph is captured.

            # `pack` will pack the args into original datatype.
            out_args, out_kwargs = self.context.inp_wrapper.pack(
                args, 
                self.context.inp_wrapper.literature, 
                self.context.inp_wrapper.nnModule
            )
            if len(out_args) == 0 and len(out_kwargs) == 0:
                out = self.inner_fn()
            elif len(out_kwargs.keys()) == 0:
                out = self.inner_fn(*out_args)
            elif len(out_args) == 0:
                out = self.inner_fn(**out_kwargs)
            else:
                out = self.inner_fn(*out_args, **out_kwargs)
            # the `out` is the original inner_fn return type. maybe not tensor.

            # `unpack` will unpack the out, record builtin datatype in self.context.out_wrapper
            # and record Tensor type in self.context.out_wrapper.tensors.
            self.context.out_wrapper.unpack(out, None)
            return self.context.out_wrapper.tensors

    # for torch.compile and do some initialize work.
    def _compile_run(self, *args):
        self._init_static_buffer()
        self._copy_to_static_args()
        self.inner_fn(*self.inp_wrapper.tensors)
        (out_args, _) = self.out_wrapper.pack(
            self.out_wrapper.tensors, 
            self.out_wrapper.literature, 
            self.out_wrapper.nnModule
        )
        return out_args

    # cudagraph need more one warmup iterations.
    def _warmup_run(self):
        if self.grad_guard and any(self.grads):
            self.inner_fn(*self.inp_wrapper.tensors)
            (out_args, _) = self.out_wrapper.pack(
                self.out_wrapper.tensors, 
                self.out_wrapper.literature, 
                self.out_wrapper.nnModule
            )
            return out_args
        else:
            self._copy_to_static_args()
            self.static_outputs_pool[self.cached_idx] = self.inner_fn(*self.static_args_pool[self.cached_idx])
            (out_args, _) = self.out_wrapper.pack(
                self.static_outputs_pool[self.cached_idx], 
                self.out_wrapper.literature, 
                self.out_wrapper.nnModule
            )
            return out_args

    def _cudagraph_run(self, graph):
        if isinstance(graph, torch.cuda.graphs.CUDAGraph):
            self._copy_to_static_args()
            graph.replay()
            (out_args, _) = self.out_wrapper.pack(
                self.static_outputs_pool[self.cached_idx], 
                self.out_wrapper.literature, 
                self.out_wrapper.nnModule
            )
            return out_args
        else:
            outputs = graph(*self.inp_wrapper.tensors)
            (out_args, _) = self.out_wrapper.pack(
                outputs, 
                self.out_wrapper.literature, 
                self.out_wrapper.nnModule
            )
            return out_args

    def _capture_run(self, idx):
        if self.grad_guard and any(self.grads):
            graph = torch.cuda.make_graphed_callables(self.inner_fn, self.inp_wrapper.tensors, allow_unused_input=True)
        else:
            graph = torch.cuda.CUDAGraph()
            self._copy_to_static_args()
            with torch.cuda.graph(graph):
                self.static_outputs_pool[self.cached_idx] = self.inner_fn(*self.static_args_pool[self.cached_idx])
        self.graph_pool[idx] = graph

    def _init_static_buffer(self):
        # cudagraph require the input tensor in a static buffer and the address unchanged during training.
        static_args = [None] * len(self.inp_wrapper.tensors)
        for i, inp in enumerate(self.inp_wrapper.tensors):
            if torch.is_floating_point(inp):
                static_args[i] = torch.randn_like(inp)
            else:
                static_args[i] = inp.clone()
        self.static_args_pool[self.cached_idx] = static_args

    def _copy_to_static_args(self):
        for i, inp in enumerate(self.inp_wrapper.tensors):
            self.static_args_pool[self.cached_idx][i].copy_(inp)

    @lru_cache(maxsize=128, typed=True)
    def runtime_cache(self, hashvalue, number, grads):
        # runtime info cache. return the item index in the cache.
        # The lru_cache key contain hashvalue, args numbers and grad info.
        # This is to avoid the hash conflict during our test experiments.
        # In the alphafold training, the `grads` is enough.
        self.func_runtimes.append(0)
        self.graph_pool.append(None)
        self.static_args_pool.append(None)
        self.static_outputs_pool.append(None)
        return len(self.func_runtimes) - 1

    def _get_runtime_info(self):
        cache_key = []
        if self.shape_guard:
            # if shape_guard is on, we will recapture the graph if the shape changes
            self.shapes = []
            for t in self.inp_wrapper.tensors:
                for i in list(t.shape):
                    self.shapes.append(i)
            cache_key = cache_key + self.shapes

        if self.grad_guard:
            # if grad_guard is on, we will recapture the graph if the requires_grad changes
            # list(self.inner_fn.parameters())[0].requires_grad is a little hack.
            self.grads = [list(self.inner_fn.parameters())[0].requires_grad and torch.is_grad_enabled()]
            cache_key = cache_key + self.grads 

        if self.literature_guard:
            # if literature_guard is on, we will recapture the graph if the python builtin type varianle changes
            self.literature = [literature for literature in self.inp_wrapper.literature]
            cache_key = cache_key + self.literature

        cache_key1 = len(cache_key)
        hashvalue = hash(tuple(cache_key))
    
        before_info = self.runtime_cache.cache_info()
        idx = self.runtime_cache(hashvalue, cache_key1, self.grads[0])
        after_info = self.runtime_cache.cache_info()
        hit = before_info.hits < after_info.hits
        return (hit, idx)

    def _run(self, *args, **kwargs):
        if not self.params_modules[0].training:
            return self.captured_object(*args, **kwargs)

        self.inp_wrapper = ParameterWrapper()

        # unpack the input parameter, record the tensor type in self.inp_wrapper.tensors
        # and other datatype in self.inp_wrapper. Can return the original datatype by call
        # inp_wrapper.pack(tensors, literatures)
        self.inp_wrapper.unpack(args, kwargs)

        hit, idx = self._get_runtime_info()
        self.cached_idx = idx

        if not hit:
            self.out_wrapper = ParameterWrapper()
            self.out_wrapper_pool.append(self.out_wrapper)
            # run torch.compile iteration. recompile if guard fail.
            return self._compile_run()
        elif self.func_runtimes[idx] < self.warmup_times:
            self.out_wrapper = self.out_wrapper_pool[idx]
            self.out_wrapper.is_record = True
            self.func_runtimes[idx] = self.func_runtimes[idx] + 1

            return self._warmup_run()
        elif self.graph_pool[idx] is None:
            self.out_wrapper = self.out_wrapper_pool[idx]
            self.out_wrapper.is_record = True

            # run cudagraph capture. recapture if guard fail.
            self._capture_run(idx)
        self.out_wrapper = self.out_wrapper_pool[idx]
        self.out_wrapper.is_record = True

        return self._cudagraph_run(self.graph_pool[idx])
 
    def __call__(self, *args, **kwargs):
        return self._run(*args, **kwargs)


class CudaGraphModuleWrapper(nn.Module):
    """CudaGraph Module Wrapper module.

    Args:
        captured_object: nn.Module.
        warmup_iterations: must be larger than 11 for ddp with cudagraph
        grad_guard: recapture if grad has changed
        shape_guard: recapture if shape has changed
        literature_guard: recapture if python variable (int, dict...) has changed
    """

    def __init__(
        self, 
        captured_object: nn.Module, 
        warmup_iterations: int = 11,
        grad_guard: bool = True,
        shape_guard: bool = True,
        literature_guard: bool = False,
    ) -> None:

        super(CudaGraphModuleWrapper, self).__init__()
        self.warmup_times = warmup_iterations
        self.literature_guard = literature_guard
        self.grad_guard = grad_guard
        self.shape_guard = shape_guard
        self.captured_object = captured_object

        self.init_inner_module = True

    def forward(self, *args, **kwargs):
        if self.init_inner_module:
            # We initialize self.context in the first iteration 
            # rather than in the __init__ method. If not, the `deepcopy`
            # will crash due to the circular reference.
            self.context = CudaGraphFunctionWrapper(
                self.captured_object, 
                [self.captured_object],
                self.warmup_times,
                self.grad_guard,
                self.shape_guard,
                self.literature_guard,
            )
            self.context.init_cudagraph()
            self.init_inner_module = False

        return self.context(*args, **kwargs)