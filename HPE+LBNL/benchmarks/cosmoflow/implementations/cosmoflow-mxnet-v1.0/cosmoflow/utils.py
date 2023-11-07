import argparse
import logging
import time
import os

from ctypes import cdll
from typing import Callable, Dict, List, Any, Optional, Tuple

import horovod.mxnet as hvd
import numpy as np
import mxnet as mx

import mxnet.cuda_utils as cu

logging.basicConfig(level=logging.WARN)


class DistributedEnvDesc(object):
    def __init__(self, local_rank: int, local_size: int, 
                 rank: int, size: int, *, 
                 mpi_library: Optional[Any] = None,
                 mpi_handler: Optional[Any] = None,
                 worker: Optional[int] = None):
        self.local_rank = local_rank
        self.local_size = local_size
        self.rank = rank
        self.size = size

        self.MPI = mpi_library
        self.comm = mpi_handler
        self.worker = worker

    @staticmethod
    def get_from_mpi(instances: int) -> 'DistributedEnvDesc':
        from mpi4py import MPI
        mpi_handler = MPI.COMM_WORLD if hasattr(MPI, "COMM_WORLD") else None

        worker = None
        if mpi_handler and instances > 1:
            assert mpi_handler.Get_size() % instances == 0, \
                f"Cannot create {instances} instances on size {mpi_handler.Get_size()} job"
            processes_per_instance = mpi_handler.Get_size() // instances
            worker = mpi_handler.Get_rank() // processes_per_instance
            mpi_handler = mpi_handler.Split(worker, mpi_handler.Get_rank())

            if worker == 0 and mpi_handler.Get_rank() == 0:
                logging.info(f"Create {instances} instances using "
                             f"{processes_per_instance} processes per instance")
        hvd.init(mpi_handler)

        #logging.info(f"Horovod initialized with: size {hvd.size()}, local_size {hvd.local_size()}")

        return DistributedEnvDesc(local_rank=hvd.local_rank(),
                                  local_size=hvd.local_size(),
                                  rank=hvd.rank(),
                                  size=hvd.size(), 
                                  mpi_library=MPI,
                                  mpi_handler=mpi_handler,
                                  worker=worker)

    @property
    def master(self) -> bool:
        return self.rank == 0

    @property
    def node(self) -> int:
        return self.rank // self.local_size


class ProfilerSection(object):
    def __init__(self, name: str, profile: bool = False):
        self.profile = profile
        self.name = name

    def __enter__(self):
        if self.profile:
            cu.nvtx_range_push(self.name)

    def __exit__(self, *args, **kwargs):
        if self.profile:
            cu.nvtx_range_pop()

libcudart = None
def cudaProfilerStart():
    global libcudart
    libcudart = cdll.LoadLibrary('libcudart.so')
    libcudart.cudaProfilerStart()

def cudaProfilerStop():
    global libcudart
    assert libcudart, "libcudart undefined or None. cudaProfilerStart should be called before cudaProfilerStop"
    libcudart.cudaProfilerStop()


def parse_cuda_profile_argument(commandline: str) -> Tuple[int, int, int]:
    temp = commandline.split(',', 1)
    temp = [temp[0]] + temp[1].split('-', 1)
    logging.debug(f"Enabled cuda profiling for range: {temp}")
    return int(temp[0]), int(temp[1]), int(temp[2])


RegisterExtensionCallbackType = Callable[[argparse.ArgumentParser], None]

class ArgumentParser(object):
    REGISTERED_EXTENSIONS: Dict[str, List[RegisterExtensionCallbackType]] = {}

    @classmethod
    def register_extension(cls, arg_group_name:str = "") -> RegisterExtensionCallbackType:
        def wrapper(function: RegisterExtensionCallbackType) -> RegisterExtensionCallbackType:
            if arg_group_name in cls.REGISTERED_EXTENSIONS:
                cls.REGISTERED_EXTENSIONS[arg_group_name].append(function)
            else:
                cls.REGISTERED_EXTENSIONS[arg_group_name] = [function]
            #logging.debug(f"Registered argument group {arg_group_name} using function {function}")
            return function
        return wrapper

    @classmethod
    def build(cls, root_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        for k, v in cls.REGISTERED_EXTENSIONS.items():
            current_parser = (root_parser if k == "" else root_parser.add_argument_group(k))
            for callback in v:
                callback(current_parser)
        return root_parser


class _LoggerBase(object):
    def register_dist_desc(self, dist: DistributedEnvDesc):
        self.dist = dist

    def _print(self, logger, key, value=None,
               metadata=None, namespace=None, stack_offset=3, uniq=True):
        if (self.dist.master and uniq) or (not uniq):
            logger(key=key, value=value, metadata=metadata, 
                   stack_offset=stack_offset)

    def _add_instance_to_kwargs(self, kwargs):
        if "metadata" not in kwargs and self.dist.worker is not None:
            kwargs["metadata"] = {"instance_id": self.dist.worker}
        elif self.dist.worker is not None:
            kwargs["metadata"]["instance_id"] = self.dist.worker

    def stop(self, *args, **kwargs):
        self.end(*args, **kwargs)

try:
    from mlperf_logging import mllog
    class Logger(_LoggerBase):
        def __init__(self, instance_id, log_template):
            logger_path = os.path.join("/results", 
                                       log_template.format(instance_id+1 if instance_id else 1))
            self.mllogger = mllog.get_mllogger()
            mllog.config(filename=logger_path)
            self.constants = mllog.constants

        def event(self, *args, **kwargs):
            self._add_instance_to_kwargs(kwargs)
            self._print(self.mllogger.event, *args, **kwargs)

        def start(self, *args, **kwargs):
            self._add_instance_to_kwargs(kwargs)
            self._print(self.mllogger.start, *args, **kwargs)

        def end(self, *args, **kwargs):
            self._add_instance_to_kwargs(kwargs)
            self._print(self.mllogger.end, *args, **kwargs)
except ImportError:
    class Logger(_LoggerBase):
        class _DummyConstant(object):
            def __getattribute__(self, name: str) -> Any:
                return name

        def __init__(self):
            self.logger = logging.getLogger(name="NVLOG")
            self.constants = self._DummyConstant()

        def _get_logger(self, event) -> Callable:
            def wrapper(key, value, metadata, stack_offset):
                logging_string = f" Encountered event of type: {event}, key={key}"
                if value is not None:
                    logging_string += f", value={value}"
                if metadata is not None:
                    logging_string += f", metadata={metadata}"
                self.logger.info(logging_string)
            return wrapper

        def event(self, *args, **kwargs):
            self._add_instance_to_kwargs(kwargs)
            self._print(self._get_logger("event"), *args, **kwargs)

        def start(self, *args, **kwargs):
            self._add_instance_to_kwargs(kwargs)
            self._print(self._get_logger("start"), *args, **kwargs)

        def end(self, *args, **kwargs):
            self._add_instance_to_kwargs(kwargs)
            self._print(self._get_logger("end"), *args, **kwargs)

logger = None



class PerformanceCounter(object):
    def __init__(self):
        self.reset()

    def update_processed(self, items: int, timestamp: int = None):
        self.last_update = timestamp if timestamp is not None else time.time()
        self.items += items

    def reset(self):
        self.items = 0
        self.start_time = time.time()
        self.last_update = self.start_time

    @property
    def throughput(self) -> float:
        return self.items / self.latency

    @property
    def latency(self) -> float:
        return self.last_update - self.start_time


class DistributedMAE(mx.metric.MAE):
    def __init__(self, dist_desc: DistributedEnvDesc, sync: bool = True, *args, **kwargs):
        self.dist_desc = dist_desc
        self.sync = sync
        super().__init__(*args, **kwargs)

    def return_global(self):
        if self.dist_desc.size > 1 and self.sync:
            total_value = np.array([self.sum_metric,
                                    self.num_inst], np.float)
            result_value = np.zeros_like(total_value)

            self.dist_desc.comm.Allreduce([total_value, self.dist_desc.MPI.FLOAT],
                                          [result_value, self.dist_desc.MPI.FLOAT],
                                          op=self.dist_desc.MPI.SUM)
            return result_value[0] / int(result_value[1])
        else:
            return self.get()[1]
