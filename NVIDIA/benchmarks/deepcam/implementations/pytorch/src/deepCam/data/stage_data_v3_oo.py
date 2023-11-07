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

import os
import io
import mmap
from glob import glob
import itertools
import numpy as np
import argparse as ap
import concurrent.futures as cf
import time
import torch.cuda.nvtx as nvtx
import copy

from mpi4py import MPI
from mpi4py.util import dtlib

import torch
import torch.distributed as dist
import io_helpers as ioh

from utils import comm

have_gds=True
try:
    import kvikio
except ImportError:
    have_gds=False
    print("Disabling GDS support")

    
def numpy_integrity_check(src_dir, dst_dir, files):
    checklist = []
    issue_found = False
    for fname in files:
        src_arr = np.load(os.path.join(src_dir, fname))
        dst_arr = np.load(os.path.join(dst_dir, fname))
        compare = np.allclose(src_arr, dst_arr,
                              rtol=1e-07,
                              atol=1e-08)
        src_nan = np.any(np.isnan(src_arr))
        dst_nan = np.any(np.isnan(dst_arr))

        issue_found = issue_found or not compare or src_nan or dst_nan

        checklist.append({"file": fname, "equal": compare, "src_nan": src_nan, "dst_nan": dst_nan})
        
    return checklist, issue_found


def get_shard(files, num_shards, shard_id, cycle_dist=0):
    num_files = len(files)
    
    # shard files into bulk and remainder:
    num_files_per_shard = num_files // num_shards
    files_bulk = files[:num_files_per_shard * num_shards]
    files_remainder = files[num_files_per_shard * num_shards:]
    
    # get my shard
    shard_start = shard_id * num_files_per_shard
    shard_end = shard_start + num_files_per_shard
    files_shard = files_bulk[shard_start:shard_end].copy()
    
    # deal with remainder: round robin with some offset for even distribution
    cycle_offset = 0
    for idf, fname in enumerate(files_remainder):
        if ((idf + cycle_offset) % num_shards == shard_id):
            files_shard.append(fname)
        cycle_offset += cycle_dist

    return files_shard  
        

def load_file(filename):
    nvtx.range_push("load_file")
    with open(filename, "rb") as f:
        token = f.read()
    nvtx.range_pop()

    return filename, token, len(token)


def load_file_direct(filename, filesize=None, blocksize=None):
    nvtx.range_push("load_file_direct")
    io_blocksize = 4096 if blocksize is None else blocksize
    token = ioh.load_file_direct(filename, blocksize=io_blocksize, filesize=0 if filesize is None else filesize)
    nvtx.range_pop()
    
    return filename, token, len(token)


def save_file(ofname, fdata):
    nvtx.range_push("save_file")
    with open(ofname, "wb") as f:
        f.write(fdata)
    nvtx.range_pop()
    return len(fdata)
    

def save_file_direct(ofname, fdata, blocksize=512, sync=True):
    nvtx.range_push("save_file_direct")
    wbytes = ioh.save_file_direct(ofname, fdata, blocksize, sync)
    nvtx.range_pop()
    return wbytes                


class FileStager(object):

    def _load_batch_parallel(self, source_directory, files, num_shards, shard_id, filesize=None, blocksize=None):

        # profiling tag
        nvtx.range_push("load_batch_parallel")
        
        # split the data across ranks
        if num_shards > 1:
            files_load = get_shard(files, num_shards, shard_id, cycle_dist=0)
        else:
            files_load = files
            
        # submit loads:
        queue = []
        for filename in files_load:
            fname = os.path.join(source_directory, filename)
            if self.use_direct_read:
                queue.append(self.read_executor.submit(load_file_direct, fname, filesize, blocksize))
            else:
                queue.append(self.read_executor.submit(load_file, fname))
        nvtx.range_pop()

        return queue

    
    def _finalize_save_local(self, save_queue):
        nvtx.range_push("finalize_save_local")
        wbytes = 0
        for handle in cf.as_completed(save_queue):
            fbytes = handle.result()
            wbytes += fbytes
        nvtx.range_pop()
        
        return wbytes

    
    # this uses the data load queue to communicate the loaded files across ranks and saves them:
    def _distribute_save_batch_parallel(self, scomm, sgroup, load_queue, target_directory):

        # profile region
        nvtx.range_push("distribute_save_batch_parallel")
        
        # we need these
        rbytes = 0
        save_queue = []
        
        # iterate till queue is empty
        for handle in cf.as_completed(load_queue):

            # load is done, extract data
            fname, fdata, fbytes = handle.result()
            
            # communicate if necessary and store
            if scomm is not None:

                # copy data to pinned buffer
                self.fdata_cpu.copy_(torch.frombuffer(fdata, dtype=torch.uint8))
                    
                # execute the gathers
                with torch.cuda.stream(self.distribute_stream):

                    # copy data to gpu tensor
                    self.fdata_gpu.copy_(self.fdata_cpu, non_blocking=True)

                    # communicate filenames
                    fname_all = scomm.allgather(fname)

                    # we need to do that to check integrity
                    if self.extended_verify:
                        from hashlib import blake2b
                        checksum = blake2b(fdata).hexdigest()
                        checksums_io = scomm.allgather(checksum) 

                    # issue allgather
                    fdata_all_gpu = [torch.empty_like(self.fdata_gpu) for _ in range(self.ssize)]
                    fdata_all_gpu[self.srank] = self.fdata_gpu
                    dist.all_gather(fdata_all_gpu, self.fdata_gpu, group=sgroup)

                    # copy back to cpu and convert to binary array
                    for ctens, gtens in zip(self.fdata_cpu_gather, fdata_all_gpu):
                        ctens.copy_(gtens, non_blocking=True)

                # sync streams
                torch.cuda.current_stream().wait(self.distribute_stream)

                # extract bytes
                fdata_all = [x.numpy().tobytes() for x in self.fdata_cpu_gather]

                # compare checksums
                if self.extended_verify:
                    checksums_ag = [blake2b(x).hexdigest() for x in fdata_all]
                    if checksums_io != checksums_ag:
                        print(checksums_io, checksums_ag)
                    assert(checksums_io == checksums_ag)
                
                for fname, fdata in zip(fname_all, fdata_all):
                    ofn = os.path.join(target_directory, os.path.basename(fname))
                    if self.use_direct_write:
                        save_queue.append(self.write_executor.submit(save_file_direct, ofn, copy.copy(fdata)))
                    else:
                        save_queue.append(self.write_executor.submit(save_file, ofn, copy.copy(fdata)))
                    rbytes += len(fdata)
            else:
                ofn = os.path.join(target_directory, os.path.basename(fname))
                if self.use_direct_write:
                    save_queue.append(self.write_executor.submit(save_file_direct, ofn, copy.copy(fdata)))
                else:
                    save_queue.append(self.write_executor.submit(save_file, ofn, copy.copy(fdata)))
                rbytes += len(fdata)
                    
        nvtx.range_pop()
                    
        return save_queue, rbytes

    
    def _stage_batch_data(self, shardinfo,
                          source_directory, target_directory,
                          files, filesize, blocksize):

        # profile region
        nvtx.range_push("stage_batch_data")
        
        # extract some shard info
        start = shardinfo["start"]
        end = shardinfo["end"]
        files_shard = files[start:end]

        # queue up the loads
        load_queue = self._load_batch_parallel(source_directory,
                                               files_shard,
                                               shardinfo["num_shards"],
                                               shardinfo["shard_id"],
                                               filesize = filesize,
                                               blocksize = blocksize)
        
        # batch loop: go one step further to process the last batch
        comm, group = (None, None) if (shardinfo["num_shards"] == 1) else (self.stage_comm, self.stage_group)
        save_queue, rbytes = self._distribute_save_batch_parallel(comm, group, load_queue, target_directory)
        
        # wait for all the stores to complete
        wbytes = self._finalize_save_local(save_queue)
        
        # close region
        nvtx.range_pop()
        
        return rbytes, wbytes
            
    
    def __init__(self,
                 global_comm,
                 num_instances,
                 instance_id,
                 instance_comm,
                 local_size,
                 local_rank,
                 batch_size = -1,
                 num_read_workers = 1,
                 num_write_workers = 1,
                 stage_mode="global",
                 verify=False,
                 full_dataset_per_node=True,
                 use_direct_io=False,
                 seed=333):

        # global chunking parameters
        self.global_comm = global_comm
        self.num_instances = num_instances
        self.instance_id = instance_id
        self.instance_comm = instance_comm
        self.local_size = local_size
        self.local_rank = local_rank

        # stage optimization info
        self.batch_size = batch_size
        self.num_read_workers = num_read_workers
        self.num_write_workers = num_write_workers
        self.stage_mode = stage_mode
        self.full_dataset_per_node = full_dataset_per_node
        self.use_direct_read = use_direct_io
        self.use_direct_write = use_direct_io
        self.seed = seed

        # set device
        self.device = torch.device(f"cuda:{self.local_rank}")

        # debug
        self.verify = verify
        # warning, this is slow!
        self.extended_verify = False

        # extract comm info
        self.gsize = self.global_comm.Get_size()
        self.grank = self.global_comm.Get_rank()
        self.isize = self.instance_comm.Get_size()
        self.irank = self.instance_comm.Get_rank()
        self.lsize = self.local_size
        self.lrank = self.local_rank

        # create helper nccl groups
        assert (dist.is_initialized()), "Error, initialize pytorch distributed first"

        # create process group from stage comm
        self.stage_comm = self.global_comm.Split(color=self.irank, key=self.instance_id)
        self.ssize = self.stage_comm.Get_size()
        self.srank = self.stage_comm.Get_rank()

        # create stage group for nccl comms
        # comm_rank = instance_rank +  instance_id * instance_size
        self.stage_group = None
        for irank in range(self.isize):
            start = irank
            end = self.gsize
            ranks = list(range(start, end, self.isize))
            tmp_group = dist.new_group(ranks = ranks)
            if irank == self.irank:
                self.stage_group = tmp_group

        # split the instance by nodes and create a comm with all matching local ranks by node
        self.num_nodes_per_instance = self.isize // self.lsize
        self.instance_node_id = self.irank // self.lsize
        
        # get a local communicator too
        self.local_comm = self.instance_comm.Split(color=(self.irank // self.lsize), key=self.lrank) 

        # create stage executor
        self.read_executor = cf.ProcessPoolExecutor(max_workers = self.num_read_workers)
        self.write_executor = cf.ProcessPoolExecutor(max_workers = self.num_write_workers)

        # create stream
        self.distribute_stream = torch.cuda.Stream()

        
    def _reset_buffers(self):
        self.fdata_cpu = None
        self.fdata_gpu = None
        self.fdata_cpu_gather = None
        

    # helper to prepare staging for the instance
    def _prepare_instance_stage(self, files, target_directory):
        if (self.stage_mode == "node") and self.full_dataset_per_node:
            files_shard = get_shard(files, self.lsize, self.lrank)
        else:
            # here we need to make sure the data is evenly distributed
            # across nodes, otherwise one node might have longer epochs
            files_shard = get_shard(files, self.isize, self.irank, cycle_dist=self.lsize)

        if self.full_dataset_per_node:
            files_print = files
        else:
            files_print = self.local_comm.allgather(files_shard)
            files_print = list(itertools.chain(*files_print))
            
        # create tags
        tag = os.path.basename(files[0]).split("-")[0]
        fname = os.path.join(target_directory, f"files_{tag}.lst")
        if self.lrank == 0:
            with open(fname, "w") as f:
                f.write("\n".join(files_print))
                
        return fname
    

    # prepare staging the instance
    def prepare(self, data_dir_prefix, stage_dir_prefix, stage_filter_list):

        # append instance ID to target dir
        target_directory = os.path.join(stage_dir_prefix, f"instance{self.instance_id}")
        
        if self.grank == 0:
            print("Copying stats.h5", flush=True)
            with open(os.path.join(data_dir_prefix, "stats.h5"), "rb") as f:
                statsfile = f.read()
        else:
            statsfile = None
            
        # broadcast the statsfile
        statsfile = self.global_comm.bcast(statsfile, 0)
            
        # save it
        if self.lrank == 0:
            os.makedirs(target_directory, exist_ok = True)  
            with open(os.path.join(target_directory, "stats.h5"), "wb") as f:
                f.write(statsfile)

        # iterate over staging filters
        self.file_stats = {}
        for stage_filter in stage_filter_list:
        
            nvtx.range_push(f"stage {stage_filter}")
        
            if (self.grank == 0):
                print(f"Preparing file lists for {stage_filter}", flush=True)
                
            # get directories
            stage_source_directory = os.path.join(data_dir_prefix, os.path.dirname(stage_filter))
            stage_target_directory = os.path.join(target_directory, os.path.dirname(stage_filter))
            
            # create target directory if not exist:
            if self.local_rank == 0:
                os.makedirs(stage_target_directory, exist_ok = True)
                
            # get file info to everybody
            if self.grank == 0:
                allfiles = sorted([os.path.basename(x) for x in glob(os.path.join(stage_source_directory, os.path.basename(stage_filter)))])
            else:
                allfiles = None
                
            # shuffle files if requested
            if (self.grank == 0) and (not self.full_dataset_per_node) and (self.seed is not None):
                rng = np.random.default_rng(self.seed)
                rng.shuffle(allfiles)

            # communicate list of files
            allfiles = self.global_comm.bcast(allfiles, 0)

            # now stage the data so that each rank in each instance has the relevant data
            stage_start = time.perf_counter()
            list_file = self._prepare_instance_stage(allfiles, stage_target_directory)
            stage_stop = time.perf_counter()
            
            # updating file stats buffer
            self.file_stats[stage_filter] = {"num_files": len(allfiles),
                                             "source_directory": stage_source_directory,
                                             "target_directory": stage_target_directory,
                                             "list_file": list_file}
            
        return

    
    def _stage_instance_data(self, stage_filter):

        # reset buffers
        self._reset_buffers()
        
        # comm parameters
        num_files = self.file_stats[stage_filter]["num_files"]
        source_directory = self.file_stats[stage_filter]["source_directory"]
        target_directory = self.file_stats[stage_filter]["target_directory"]
        
        # this is the tag of the file list
        fname = self.file_stats[stage_filter]["list_file"]

        # load metadata
        with open(fname, "r") as f:
            files_shard = f.read().splitlines()

        # shard locally
        files_shard = get_shard(files_shard, self.lsize, self.lrank)

        # get file stats:
        stats = os.stat(os.path.join(source_directory, files_shard[0]))
        filesize = stats.st_size
        blocksize = stats.st_blksize
        self.file_stats[stage_filter]["filesize"] = filesize
        self.file_stats[stage_filter]["blocksize"] = blocksize

        # init buffers
        if self.fdata_cpu is None:
            self.fdata_cpu = torch.empty(len(fdata), dtype=torch.uint8).pin_memory()
            
        if self.fdata_cpu_gather is None:
            self.fdata_cpu_gather = [torch.empty_like(self.fdata_cpu).pin_memory() for _ in range(self.ssize)]

        if self.fdata_gpu is None:
            self.fdata_gpu = torch.empty_like(self.fdata_cpu, device=self.device)
        

        # now, let's take care of the data: update the number of files because of remainder
        batch_size = self.batch_size
        if self.stage_mode == "global":
            batch_size *= self.ssize
        num_files_per_shard = len(files_shard)
        num_batches_bulk = num_files_per_shard // batch_size
        num_files_remainder = num_files_per_shard - num_batches_bulk * batch_size
                    
        # create list of batch sizes, shard sizes, etc:
        stage_info = [{"start": i*batch_size,
                       "end": (i+1)*batch_size,
                       "num_shards": self.ssize if self.stage_mode == "global" else 1,
                       "shard_id": self.srank if self.stage_mode == "global" else 0}
                      for i in range(0, num_batches_bulk)]
        
        # deal with the remainder:
        remainder_start = num_batches_bulk * batch_size
        if self.stage_mode == "global":
            # see if we can squeeze in one more batch with reduced size
            eff_batchsize = (num_files_remainder // self.ssize) * self.ssize
            
            if eff_batchsize > 0:
                stage_info.append({"start": remainder_start,
                                   "end": remainder_start + eff_batchsize,
                                   "num_shards": self.ssize,
                                   "shard_id": self.srank})
                
            remainder_start += eff_batchsize

        # remainder:
        if (num_files_per_shard - remainder_start > 0):
            stage_info.append({"start": remainder_start,
                               "end": num_files_per_shard,
                               "num_shards": 1,
                               "shard_id": 0})

        # do the staging
        total_bytes_read = 0
        total_bytes_write = 0
        for shardinfo in stage_info:
            rbytes, wbytes = self._stage_batch_data(shardinfo,
                                                    source_directory,
                                                    target_directory,
                                                    files_shard,
                                                    filesize, blocksize)
            total_bytes_read += rbytes
            total_bytes_write += wbytes
            
        # global barrier
        self.instance_comm.Barrier()
        
        return total_bytes_read, total_bytes_write

    
    def execute_stage(self):

        # some useful variables
        # unit conversion
        unit_convert_gb = 1./float(1024*1024*1024)        
        
        # iterate over all the prepared stage filters
        for stage_filter in self.file_stats.keys():

            nvtx.range_push(f"stage {stage_filter}")

            stage_start = time.perf_counter()
            total_read, total_write = self._stage_instance_data(stage_filter)
            stage_stop = time.perf_counter()

            # allreduce:
            total_read = self.global_comm.allreduce(total_read)
            total_write = self.global_comm.allreduce(total_write)

            # convert units
            total_read *= unit_convert_gb
            total_write *= unit_convert_gb
            
            # stage duration:
            stage_duration = stage_stop - stage_start
            
            # print
            if self.grank == 0:
                print(f"""Staging {stage_filter} done.
                          Total number of files: {self.file_stats[stage_filter]["num_files"]}.
                          Elapsed time {stage_duration:.2f}s.
                          Read {total_read:.2f} GB (bandwidth: {total_read/stage_duration:.2f} GB/s).
                          Write {total_write:.2f} GB (bandwidth: {total_write/stage_duration:.2f} GB/s).
                       """, flush=True)

            # verify staging results if requested
            if self.verify:
                nvtx.range_push(f"stage_verify")
                if self.lrank == 0:
                    stage_target_directory = self.file_stats[stage_filter]["target_directory"]
                    staged_files = glob(os.path.join(stage_target_directory, os.path.basename(stage_filter)))
                else:
                    staged_files = []
                    
                if not self.full_dataset_per_node:
                    # if every node hosts a shard, we need to sum the results, if not we need to make sure everybody has the same
                    staged_files_full = self.instance_comm.allgather(staged_files)
                    staged_files_full = set(itertools.chain(*staged_files_full))
                else:
                    staged_files_full = set(staged_files)
                staged_num_files = len(staged_files_full)

                global_files_full = None
                if self.irank == 0:
                    stage_source_directory = self.file_stats[stage_filter]["source_directory"]
                    global_files_full = glob(os.path.join(stage_source_directory, os.path.basename(stage_filter)))
                global_files_full = self.instance_comm.bcast(global_files_full, 0)
                global_num_files = len(global_files_full)
                        
                # strip off the directory
                checkfiles1 = sorted([os.path.basename(x) for x in staged_files_full])
                checkfiles2 = sorted([os.path.basename(x) for x in global_files_full])

                assert(staged_num_files == global_num_files), f"Error, expected {global_num_files} but got {staged_num_files}"
                assert(checkfiles1 == checkfiles2)

                if self.extended_verify:
                    if self.lrank == 0:
                        checks, issue = numpy_integrity_check(self.file_stats[stage_filter]["source_directory"],
                                                              self.file_stats[stage_filter]["target_directory"],
                                                              [os.path.basename(x) for x in staged_files])
                        if issue:
                            print(f"Instance {self.instance_id}, local rank {self.irank}: Verification Error. Results:", checks, flush=True)
                        else:
                            print(f"Instance {self.instance_id}, local rank {self.irank}: Verification OK", flush=True)
                    self.instance_comm.Barrier()
                        
                if self.irank == 0:
                    print(f'Staged data for {stage_filter}: {staged_num_files}, expected: {global_num_files}', flush=True)
                nvtx.range_pop()

            # close range
            nvtx.range_pop()
            
        return

    
