import os
import mmap
from glob import glob
import itertools
import numpy as np
import argparse as ap
import concurrent.futures as cf
import time
from queue import Queue as Queue
import torch.cuda.nvtx as nvtx
import copy

import torch
import io_helpers as ioh


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


def exchange_buffers(comm, fdata):
    # quick exit if we do not need to communicate
    if comm.Get_size() == 1:
        rbytes = sum([len(x[1]) for x in fdata])
        return fdata, rbytes
    
    # profile region start
    nvtx.range_push("exchange_buffers")

    # gather
    fdata_all = comm.allgather(fdata)
    
    # flatten
    fdata_result = list(itertools.chain(*fdata_all))
    
    # size:
    rbytes = sum([len(x[1]) for x in fdata_result])
    
    # stop profiling
    nvtx.range_pop()
    
    # return
    return fdata_result, rbytes                                                                


def load_file(filename):
    with open(filename, "rb") as f:
        token = f.read()

    return filename, token, len(token)


def load_file_direct(filename, filesize=None, blocksize=None):
    io_blocksize = 4096 if blocksize is None else blocksize
    token = ioh.load_file_direct(filename, blocksize=io_blocksize, filesize=0 if filesize is None else filesize)
    
    return filename, token, len(token)


def save_file(ofname, fdata):
    with open(ofname, "wb") as f:
        f.write(fdata)
    return len(fdata)
                    

def save_file_direct(ofname, fdata, blocksize=512):
    wbytes = ioh.save_file_direct(ofname, fdata, blocksize)
    return wbytes 


class FileStager(object):
    
    def _finalize_load_local(self, fdata_handles):
        nvtx.range_push("finalize_load_local")
        file_data = []
        rbytes = 0
        while not fdata_handles.empty():
            handle = fdata_handles.get()
            if handle.done():
                fname, fdata, fbytes = handle.result()
                file_data.append((fname, fdata))
                rbytes += fbytes
            else:
                fdata_handles.put(handle)
        nvtx.range_pop()
        
        return file_data, rbytes 


    def _finalize_load_global(self, files, rbytes):
        nvtx.range_push("finalize_load_global")
        files_all, rbytes = exchange_buffers(self.stage_comm, files)
        #we can compute that from the gathered data, no need to sync up
        #rbytes = comm.allreduce(rbytes)
        nvtx.range_pop()
        
        return files_all, rbytes


    def _load_batch_parallel(self, source_directory, files, filesize=None, blocksize=None):
        nvtx.range_push("load_batch_parallel")

        # shard info
        num_shards = self.ssize if self.stage_mode == "global" else 1
        shard_id = self.srank if self.stage_mode == "global" else 0
        
        # split the data across ranks
        if num_shards > 1:
            files_load = get_shard(files, num_shards, shard_id, cycle_dist=0)
        else:
            files_load = files

        # submit loads:
        queue = Queue()
        for filename in files_load:
            fname = os.path.join(source_directory, filename)
            if self.use_direct_io:
                queue.put(self.executor.submit(load_file_direct, fname, filesize, blocksize))
            else:
                queue.put(self.executor.submit(load_file, fname))
        nvtx.range_pop()

        return queue
    
    
    def _save_batch_parallel(self, target_directory, fdata):
        nvtx.range_push("save_batch_parallel")
        queue = Queue()
        for fn, fd in fdata:
            ofn = os.path.join(target_directory, os.path.basename(fn))
            if self.use_direct_io:
                queue.put(self.executor.submit(save_file_direct, ofn, copy.copy(fd)))
            else:
                queue.put(self.executor.submit(save_file, ofn, copy.copy(fd)))
        nvtx.range_pop()

        return queue                                            


    def _finalize_save_local(self, save_queue):
        nvtx.range_push("finalize_save_local")
        wbytes = 0
        while not save_queue.empty():
            handle = save_queue.get()
            if handle.done():
                fbytes = handle.result()
                wbytes += fbytes
            else:
                save_queue.put(handle)
        nvtx.range_pop()

        return wbytes                                                                                      
        
    
    def __init__(self,
                 global_comm,
                 num_instances,
                 instance_id,
                 instance_comm,
                 local_size,
                 local_rank,
                 batch_size = -1,
                 num_workers = 1,
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
        self.num_workers = num_workers                        
        self.stage_mode = stage_mode
        self.verify = verify
        self.full_dataset_per_node = full_dataset_per_node
        self.use_direct_io = use_direct_io
        self.seed = seed

        # extract comm info
        self.gsize = self.global_comm.Get_size()
        self.grank = self.global_comm.Get_rank()
        self.isize = self.instance_comm.Get_size()
        self.irank = self.instance_comm.Get_rank()
        self.lsize = self.local_size
        self.lrank = self.local_rank

        # create new helper comms
        self.stage_comm = self.global_comm.Split(color=self.irank, key=self.instance_id)
        self.ssize = self.stage_comm.Get_size()
        self.srank = self.stage_comm.Get_rank()
        # split the instance by nodes and create a comm with all matching local ranks by node
        self.num_nodes_per_instance = self.isize // self.lsize
        self.instance_node_id = self.irank // self.lsize
        self.instance_node_comm = self.instance_comm.Split(color=self.lrank, key=self.instance_node_id)
        # get a local communicator too
        self.local_comm = self.instance_comm.Split(color=(self.irank // self.lsize), key=self.lrank)

        # create stage executor
        self.executor = cf.ThreadPoolExecutor(max_workers = self.num_workers)


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
        tag = files[0].split("-")[0]
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

        # now, let's take care of the data: update the number of files because of remainder
        num_files_per_shard = len(files_shard)
        if self.batch_size > 0:
            num_batches = (num_files_per_shard + self.batch_size - 1) // self.batch_size
        else:
            num_batches = 1
            batch_size = num_files_per_shard
            
        # get file sizes:
        stats = os.stat(os.path.join(source_directory, files_shard[0]))
        filesize = stats.st_size
        blocksize = stats.st_blksize
        self.file_stats[stage_filter]["filesize"] = filesize
        self.file_stats[stage_filter]["blocksize"] = blocksize

        # create staging pipeline
        fdata_handles = self._load_batch_parallel(source_directory,
                                                  files_shard[0:min(num_files_per_shard, self.batch_size)],
                                                  filesize=filesize,
                                                  blocksize=blocksize)                                                  

        # batch loop: go one step further to process the last batch
        save_handles = []
        total_bytes_read = 0
        total_bytes_write = 0
        for bid in range(1, num_batches+1):
            # wait for data loader
            fdata_save, rbytes = self._finalize_load_local(fdata_handles)
            
            # finalize the load if required:
            if self.stage_mode == "global":
                fdata_save, rbytes = self._finalize_load_global(fdata_save, rbytes)
                
            # increment total bytes read
            total_bytes_read += rbytes

            # get next batch, only submit if there are any
            if bid < num_batches:
                batch_start = bid * self.batch_size
                batch_end = min(batch_start + self.batch_size, num_files_per_shard)
                fdata_handles = self._load_batch_parallel(source_directory,
                                                          files_shard[batch_start:batch_end],
                                                          filesize=filesize,
                                                          blocksize=blocksize)

            # exchange buffers inside instance if necessary
            if self.full_dataset_per_node and (self.stage_mode != "node"):
                fdata_save, _ = exchange_buffers(self.instance_node_comm, fdata_save)

            # wait till data is saved
            if save_handles:
                total_bytes_write += self._finalize_save_local(save_handles)

            # store locally
            save_handles = self._save_batch_parallel(target_directory, fdata_save)

        if save_handles:
            total_bytes_write += self._finalize_save_local(save_handles)

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
                       """)

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

                assert(staged_num_files == global_num_files)
                assert(checkfiles1 == checkfiles2)
                        
                if self.irank == 0:
                    print(f'Staged data for {stage_filter}: {staged_num_files}, expected: {global_num_files}', flush=True)
                nvtx.range_pop()

            # close range
            nvtx.range_pop()
            
        return
