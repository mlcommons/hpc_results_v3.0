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

import utils.libCosmoflowExt as cfext
import abc
import concurrent.futures
import functools
import pathlib
import os

import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib

from typing import Iterable, Callable, Tuple, Optional

from omegaconf import OmegaConf
from utils import DistributedEnv, ExecutionTimer

import gzip
import multiprocessing as mp


def allgather_safe(comm, fdata):
    # total size
    comm_size = comm.Get_size()
    num_bytes = len(fdata)
    total_bytes = num_bytes * comm_size

    # chunk by ~1GB:
    gigabyte = 1024*1024*1024

    # determine number of chunks
    num_chunks = (total_bytes + gigabyte - 1) // gigabyte

    # determine local chunksize
    chunksize = (num_bytes + num_chunks - 1) // num_chunks

    # datatype stuff
    datatype = MPI.BYTE
    np_dtype = dtlib.to_numpy_dtype(datatype)

    # gather stuff
    if True:
        # prepare buffers:
        sendbuff = np.frombuffer(memoryview(
            fdata), dtype=np_dtype, count=num_bytes)
        recvbuff = np.empty((comm_size * chunksize), dtype=np_dtype)
        resultbuffs = np.split(
            np.empty(num_bytes * comm_size, dtype=np_dtype), comm_size)

        # do subsequent gathers
        for i in range(0, num_chunks):
            # create buffer views
            start = i * chunksize
            end = min(start + chunksize, num_bytes)
            eff_bytes = end - start
            sendbuffv = sendbuff[start:end]
            recvbuffv = recvbuff[0:eff_bytes*comm_size]

            # perform allgather on views
            comm.Allgather([sendbuffv, datatype], [recvbuffv, datatype])

            # split result buffer for easier processing
            recvbuff_split = np.split(recvbuffv, comm_size)
            for j in range(comm_size):
                resultbuffs[j][start:end] = recvbuff_split[j][...]
        results = [x.tobytes() for x in resultbuffs]
    else:
        recvbuff = np.empty((total_bytes), dtype=np_dtype)
        for i in range(0, num_chunks):
            # prepare local chunks
            local_start = i * chunksize
            local_end = min(local_start + chunksize, num_bytes)
            eff_bytes = local_end - local_start

            # set up send buffer and recv buffer specv
            sendbuff = np.frombuffer(memoryview(
                fdata[local_start:local_end]), dtype=np_dtype, count=eff_bytes)
            counts = [eff_bytes for _ in range(comm_size)]
            recv_displacements = [local_start + j *
                                  num_bytes for j in range(comm_size)]

            # perform the gather
            comm.Allgatherv([sendbuff, datatype], [
                            recvbuff, (counts, recv_displacements), datatype])

        # create the output vector
        recvbuff_split = np.split(recvbuff, comm_size)
        results = [x.tobytes() for x in recvbuff_split]

    return results


def save_idx_file(data: bytes, output: str):
    import struct
    with open(output, "w") as f:
        current = 0
        while current < len(data):
            proto_len = struct.unpack('q', data[current:current+8])[0]
            f.write(str(current) + ' ' + str(16 + proto_len) + '\n')
            current += 16 + proto_len


class AbstractStagerExecutor(object):
    def __enter__(self) -> "AbstractStagerExecutor":
        return self

    def __exit__(self, error_type, error, traceback) -> None:
        pass

    @abc.abstractmethod
    def stage_files(self,
                    data_sample_list: Iterable[str],
                    data_target_list: Iterable[str],
                    input_dir: pathlib.Path,
                    output_dir: pathlib.Path,
                    *,
                    profile: bool = False,
                    compressed: bool = False,
                    size: Optional[np.ndarray] = None) -> Callable[[], bool]:
        pass

    @classmethod
    def _stage_single_example(cls,
                              data_sample: str,
                              data_target: str,
                              *,
                              input_dir: pathlib.Path,
                              output_dir: pathlib.Path,
                              profile: bool = False,
                              compressed: bool = False) -> None:
        def copy_single_and_verify(input_path: pathlib.Path,
                                   output_path: pathlib.Path) -> None:
            with open(output_path, "wb") as fdst:
                if compressed:
                    from utils.libCosmoflowExt import read_file
                    data = read_file(str(input_path), 4096, 0)
                else:
                    with open(input_path, "rb") as fsrc:
                        data = fsrc.read()
                        data = gzip.decompress(data)

                if str(output_path).endswith(".tfrecord"):
                    save_idx_file(data, f"{str(output_path)}.idx")
                fdst.write(data)

        with ExecutionTimer(f"stage_{data_sample}", profile=profile):
            copy_single_and_verify(input_dir / data_sample,
                                   output_dir / data_sample)
            if data_target is not None:
                copy_single_and_verify(input_dir / data_target,
                                       output_dir / data_target)


class SequentialStagerExecutor(AbstractStagerExecutor):
    def stage_files(self,
                    data_sample_list: Iterable[str],
                    data_target_list: Iterable[str],
                    input_dir: pathlib.Path,
                    output_dir: pathlib.Path,
                    *,
                    profile: bool = False,
                    compressed: bool = False):
        for data, label in zip(data_sample_list,
                               data_target_list):
            SequentialStagerExecutor._stage_single_example(
                data, label,
                input_dir=input_dir,
                output_dir=output_dir,
                profile=profile,
                compressed=compressed)

        return lambda: True


class ThreadPoolStagerExecutor(AbstractStagerExecutor):
    def __init__(self, max_workers: int = 4):
        self.thread_pool_workers = max_workers
        self.thread_pool = None

    def __enter__(self) -> AbstractStagerExecutor:
        if self.thread_pool is None:
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                self.thread_pool_workers).__enter__()
        return self

    def __exit__(self, err_type, err, traceback):
        if self.thread_pool is not None:
            result = self.thread_pool.__exit__(err_type, err, traceback)
            self.thread_pool = None
        return result

    def stage_files(self,
                    data_sample_list: Iterable[str],
                    data_target_list: Iterable[str],
                    input_dir: pathlib.Path,
                    output_dir: pathlib.Path,
                    *,
                    profile: bool = False,
                    compressed: bool = False) -> Callable[[], None]:
        if data_target_list is None:
            result_iter = self.thread_pool.map(functools.partial(self._stage_single_example,
                                                                 data_target=None,
                                                                 input_dir=input_dir,
                                                                 output_dir=output_dir,
                                                                 profile=profile,
                                                                 compressed=compressed),
                                               data_sample_list)
        else:
            result_iter = self.thread_pool.map(functools.partial(self._stage_single_example,
                                                                 input_dir=input_dir,
                                                                 output_dir=output_dir,
                                                                 profile=profile,
                                                                 compressed=compressed),
                                               data_sample_list, data_target_list)

        def _wait_till_finish():
            for _ in result_iter:
                pass
        return _wait_till_finish


class ProcessPoolStagerExecutor(AbstractStagerExecutor):
    def __init__(self, max_workers: int = 4):
        self.thread_pool_workers = max_workers
        self.thread_pool = None

    def __enter__(self) -> AbstractStagerExecutor:
        if self.thread_pool is None:
            self.thread_pool = mp.Pool(
                self.thread_pool_workers).__enter__()
        return self

    def __exit__(self, err_type, err, traceback):
        if self.thread_pool is not None:
            result = self.thread_pool.__exit__(err_type, err, traceback)
            self.thread_pool = None
        return result

    def stage_files(self,
                    data_sample_list: Iterable[str],
                    data_target_list: Iterable[str],
                    input_dir: pathlib.Path,
                    output_dir: pathlib.Path,
                    *,
                    profile: bool = False,
                    compressed: bool = False,
                    **kwargs) -> Callable[[], None]:
        if data_target_list is None:
            result_iter = self.thread_pool.map(functools.partial(self._stage_single_example,
                                                                 data_target=None,
                                                                 input_dir=input_dir,
                                                                 output_dir=output_dir,
                                                                 profile=profile,
                                                                 compressed=compressed),
                                               data_sample_list)
        else:
            result_iter = self.thread_pool.starmap(functools.partial(self._stage_single_example,
                                                                     input_dir=input_dir,
                                                                     output_dir=output_dir,
                                                                     profile=profile,
                                                                     compressed=compressed),
                                                   zip(data_sample_list, data_target_list))

        def _wait_till_finish():
            for _ in result_iter:
                pass
        return _wait_till_finish


class ThreadPoolDirectExecutor(AbstractStagerExecutor):
    def __init__(self,
                 distenv: DistributedEnv,
                 max_workers: int = 4):
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers)
        self._distenv = distenv

        if self._distenv.num_instances > 1:
            self._dist_comm = self._distenv.master_mpi_comm.Split(
                color=self._distenv.instance_mpi_comm.Get_rank(),
                key=self._distenv.master_mpi_comm.Get_rank())

            assert self._dist_comm.Get_size() == self._distenv.num_instances, \
                "Something went wrong during distributed stager initialization"
        else:
            self._dist_comm = None

    def __enter__(self) -> "AbstractStagerExecutor":
        if self._thread_pool is not None:
            return self._thread_pool.__enter__()
        return None

    def __exit__(self, error_type, error, traceback) -> None:
        if self._thread_pool is not None:
            return self._thread_pool.__exit__(error_type, error, traceback)
        return None

    def await_and_return(self, queue):
        concurrent.futures.wait(
            queue, return_when=concurrent.futures.ALL_COMPLETED)

    def _non_distributed_stage(self,
                               data_sample_list: Iterable[str],
                               sizes: Optional[np.ndarray],
                               input_dir: pathlib.Path,
                               output_dir: pathlib.Path,
                               *,
                               profile: bool = False) -> Callable[[], None]:
        fused_futures = []

        print(f"Staging non distributed files ({len(data_sample_list)})")

        with ExecutionTimer(f"schedule_thread_read_non_distributed", profile=profile):
            for data_sample, size in zip(data_sample_list, sizes):
                future = self._thread_pool.submit(cfext.read_write_fused,
                                                  str(input_dir / data_sample),
                                                  str(output_dir / data_sample),
                                                  int(size), 4096, True)
                fused_futures.append(future)

        return lambda: self.await_and_return(fused_futures)

    def stage_files(self,
                    data_sample_list: Iterable[str],
                    data_target_list: Iterable[str],
                    input_dir: pathlib.Path,
                    output_dir: pathlib.Path,
                    *,
                    profile: bool = False,
                    compressed: bool = False,
                    sizes: Optional[np.ndarray] = None) -> Callable[[], bool]:
        assert data_target_list is None, "ThreadPoolDirectExecutor can bu used only with packed data samples"

        total_files_to_stage = len(data_sample_list)
        if self._dist_comm is not None:
            no_dist_workers = self._dist_comm.Get_size()
            total_chunks_to_sync = (total_files_to_stage // no_dist_workers)
            total_remain_to_sync = total_files_to_stage - \
                (total_chunks_to_sync * no_dist_workers)
        else:
            total_remain_to_sync = total_files_to_stage
            total_chunks_to_sync = 0

        load_futures = []
        save_futures = []

        np_dtype = dtlib.to_numpy_dtype(MPI.BYTE)

        # print(
        #    f"Distributed stager: chunk syncs ({total_chunks_to_sync}), non distributed sync ({total_remain_to_sync})")

        dist_awaiter = None
        if total_chunks_to_sync > 0:
            dist_rank = self._dist_comm.Get_rank()
            dist_size = self._dist_comm.Get_size()
            batch_size = min(dist_size, 4)

            for file_name, file_size in zip(data_sample_list[dist_rank * total_chunks_to_sync:(dist_rank+1) * total_chunks_to_sync],
                                            sizes[dist_rank * total_chunks_to_sync:(dist_rank+1) * total_chunks_to_sync]):
                future = self._thread_pool.submit(
                    ThreadPoolDirectExecutor.read_worker_func, str(input_dir / file_name), 4096, int(file_size))
                load_futures.append(future)

            chunk = 0
            for load_future in concurrent.futures.as_completed(load_futures):
                name, data = load_future.result()
                FILE_SIZE_CONSTANT = 16777287

                names = self._dist_comm.allgather(os.path.basename(name))
                assert len(data) == FILE_SIZE_CONSTANT, \
                    "Data size didn't match the expected value!"
                assert len(names) == dist_size, \
                    "File list received size didn't match the dist_size"

                chunk += 1
                #recvbuff = np.empty((dist_size * FILE_SIZE_CONSTANT),
                #                    dtype=np_dtype)
                #sendbuff = np.frombuffer(memoryview(data),
                #                         dtype=np_dtype)

                #self._dist_comm.Allgather(
                #    [sendbuff, MPI.BYTE], [recvbuff, MPI.BYTE])

                file_data_list = allgather_safe(self._dist_comm, data)

                file_name_list = [str(output_dir / name) for name in names]
                #file_data_list = [recvbuff[i*FILE_SIZE_CONSTANT:(i+1)*FILE_SIZE_CONSTANT].tobytes()
                #                  for i in range(dist_size)]

                save_futures.append(self._thread_pool.submit(cfext.write_file_batch,
                                                             file_name_list,
                                                             file_data_list,
                                                             4096,
                                                             True))

                queue_size = len(save_futures) * FILE_SIZE_CONSTANT * dist_size
                if queue_size > 64 * 1024 * 1024 * 1024:
                    result = concurrent.futures.wait(save_futures,
                                                     return_when=concurrent.futures.FIRST_COMPLETED)
                    save_futures = list(result.not_done)

            def dist_awaiter(): return self.await_and_return(save_futures)

        if total_remain_to_sync > 0:
            start_idx = total_files_to_stage - total_remain_to_sync
            awaiter = self._non_distributed_stage(data_sample_list[start_idx:],
                                                  sizes[start_idx:],
                                                  input_dir=input_dir,
                                                  output_dir=output_dir,
                                                  profile=profile)
        else:
            awaiter = None
        awaiter = None

        def final_awaiter():
            if dist_awaiter is not None:
                dist_awaiter()
            if awaiter is not None:
                awaiter()
        return final_awaiter

    @ staticmethod
    def read_worker_func(name: str, blk: int, size: int) -> Tuple[str, bytes]:
        return name, cfext.read_file(name, blk, size)


def get_executor_from_config(distenv: DistributedEnv, config: OmegaConf) -> AbstractStagerExecutor:
    executor_type = config["data"]["stage_mode"]

    if executor_type == "sequential":
        return SequentialStagerExecutor()
    else:
        executor_workers = config["data"]["stage_workers"]
        if executor_type == "thread":
            return ThreadPoolDirectExecutor(distenv, executor_workers)
        elif executor_type == "process":
            return ProcessPoolStagerExecutor(executor_workers)
        else:
            raise RuntimeError(
                "Invalid executor type for staging. Use a proper one")
