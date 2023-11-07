import os
import io
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

from mpi4py import MPI
from mpi4py.util import dtlib

import torch
import io_helpers as ioh

# small sharding helper function
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
    with open(filename, "rb") as f:
        token = f.read()

    return filename, token, len(token)


def load_file_direct(filename, filesize=None, blocksize=None):

    # set blocksize and load
    io_blocksize = 4096 if blocksize is None else blocksize
    token = ioh.load_file_direct(filename, blocksize=io_blocksize, filesize=0 if filesize is None else filesize)
    
    return filename, token, len(token)


def load_batch_parallel(executor, files, comm_size, comm_rank, filesize=None, blocksize=None, direct_io=False):
    nvtx.range_push("load_batch_parallel")
    # split the data across ranks
    if comm_size > 1:
        files_load = get_shard(files, comm_size, comm_rank, cycle_dist=0)
    else:
        files_load = files

    # submit loads:
    queue = Queue()
    for filename in files_load:
        if direct_io:
            queue.put(executor.submit(load_file_direct, filename, filesize, blocksize))
        else:
            queue.put(executor.submit(load_file, filename))
    nvtx.range_pop()
    
    return queue


def finalize_save_local(save_queue):
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


def save_file(ofname, fdata):
    with open(ofname, "wb") as f:
        f.write(fdata)
    return len(fdata)


def save_file_direct(ofname, fdata, blocksize=512):
    wbytes = ioh.save_file_direct(ofname, fdata, blocksize)
    return wbytes

    
def allgather_safe(comm, fdata):
    #total size
    comm_size = comm.Get_size()
    num_bytes = len(fdata)
    total_bytes = num_bytes * comm_size

    #chunk by ~1GB:
    gigabyte = 1024*1024*1024
    #gigabyte = 1024 * 1024

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
        sendbuff = np.frombuffer(memoryview(fdata), dtype=np_dtype, count=num_bytes)
        recvbuff = np.empty((comm_size * chunksize), dtype=np_dtype)
        resultbuffs = np.split(np.empty(num_bytes * comm_size, dtype=np_dtype), comm_size)

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
            sendbuff = np.frombuffer(memoryview(fdata[local_start:local_end]), dtype=np_dtype, count=eff_bytes)
            counts = [eff_bytes for _ in range(comm_size)]
            recv_displacements = [local_start + j * num_bytes for j in range(comm_size)]
            
            # perform the gather
            comm.Allgatherv([sendbuff, datatype], [recvbuff, (counts, recv_displacements), datatype])

        # create the output vector
        recvbuff_split = np.split(recvbuff, comm_size)
        results = [x.tobytes() for x in recvbuff_split]

    return results


# this uses the data load queue to communicate the loaded files across ranks and saves them:
def distribute_save_batch_parallel(comm, executor, load_queue, target_directory):

    # profile region
    nvtx.range_push("distribute_save_batch_parallel")
    
    # we need these
    rbytes = 0
    save_queue = Queue()

    # iterate till queue is empty
    while not load_queue.empty():
        
        # get handle
        handle = load_queue.get()

        # if load is not done, requeue
        if not handle.done():
            load_queue.put(handle)
            continue

        # load is done, extract data
        fname, fdata, fbytes = handle.result()

        # communicate if necessary and store
        if comm is not None:
            fname_all = comm.allgather(fname)
            fdata_all = allgather_safe(comm, fdata)
            for fname, fdata in zip(fname_all, fdata_all):
                ofn = os.path.join(target_directory, os.path.basename(fname))
                save_queue.put(executor.submit(save_file, ofn, copy.copy(fdata)))
                rbytes += len(fdata)
        else:
            rbytes += len(fdata)
            ofn = os.path.join(target_directory, os.path.basename(fname))
            save_queue.put(executor.submit(save_file, ofn, copy.copy(fdata)))

    nvtx.range_pop()
            
    return save_queue, rbytes


# this routine stages one batch of a time, overlapping communication and loads and stores
def stage_batch_data(stage_comm, executor, shardinfo, files, filesize, blocksize, target_directory, use_direct_io):
    
    # profile region
    nvtx.range_push("stage_batch_data")
    
    # extract some shard info
    start = shardinfo["start"]
    end = shardinfo["end"]
    files_shard = files[start:end]

    # queue up the loads
    load_queue = load_batch_parallel(executor, files_shard,
                                     shardinfo["num_shards"],
                                     shardinfo["shard_id"],
                                     filesize = filesize,
                                     blocksize = blocksize,
                                     direct_io = use_direct_io)

    # batch loop: go one step further to process the last batch
    comm = None if (shardinfo["num_shards"] == 1) else stage_comm
    save_queue, rbytes = distribute_save_batch_parallel(comm, executor, load_queue, target_directory)

    # wait for all the stores to complete
    wbytes = finalize_save_local(save_queue)

    # close region
    nvtx.range_pop()
    
    return rbytes, wbytes


# this routine stages data for each instance
def stage_instance_data(stage_comm, instance_comm, instance_node_comm,
                        lsize, lrank,
                        files, target_directory,
                        batch_size=-1,
                        stage_num_workers=1,
                        stage_mode="node",
                        full_dataset_per_node=True,
                        use_direct_io=False,
                        prepare_staging=False):

    # comm parameters
    ssize = stage_comm.Get_size()
    srank = stage_comm.Get_rank()
    isize = instance_comm.Get_size()
    irank = instance_comm.Get_rank()
    nsize = instance_node_comm.Get_size()
    nrank = instance_node_comm.Get_rank()

    num_files = len(files)


    tag = os.path.basename(files[0]).split("-")[0]
    fname = os.path.join(target_directory, f"files_{tag}.lst")
    if prepare_staging:

        if (stage_mode == "node") and full_dataset_per_node:
            files_shard = get_shard(files, lsize, lrank)
        else:
            # here we need to make sure the data is evenly distributed
            # across nodes, otherwise one node might have longer epochs
            files_shard = get_shard(files, isize, irank, cycle_dist=lsize)
            
        if full_dataset_per_node:
            files_print = files
        else:
            local_comm = instance_comm.Split(color=(irank // lsize), key=lrank)
            files_print = local_comm.allgather(files_shard)
            files_print = list(itertools.chain(*files_print))
        
        # create tags
        if lrank == 0:
            with open(fname, "w") as f:
                f.write("\n".join(files_print))
        return 0, 0

    else:
        # load metadata
        with open(fname, "r") as f:
            files_shard = f.read().splitlines()

        # shard locally
        files_shard = get_shard(files_shard, lsize, lrank)
    
    # automatic batch size adjustment:
    if stage_mode == "global":
        batch_size *= ssize
    
    # now, let's take care of the data: update the number of files because of remainder
    num_files_per_shard = len(files_shard)
    num_batches_bulk = num_files_per_shard // batch_size
    num_files_remainder = num_files_per_shard - num_batches_bulk * batch_size
        
    # create executor
    executor = cf.ThreadPoolExecutor(max_workers = stage_num_workers)

    # get file sizes:
    stats = os.stat(files_shard[0])
    filesize = stats.st_size
    blocksize = stats.st_blksize

    # create list of batch sizes, shard sizes, etc:
    stage_info = [{"start": i*batch_size,
                   "end": (i+1)*batch_size,
                   "num_shards": ssize if stage_mode == "global" else 1,
                   "shard_id": srank if stage_mode == "global" else 0}
                  for i in range(0, num_batches_bulk)]

    # deal with the remainder:
    remainder_start = num_batches_bulk * batch_size
    if stage_mode == "global":
        # see if we can squeeze in one more batch with reduced size
        eff_batchsize = (num_files_remainder // ssize) * ssize
        
        if eff_batchsize > 0:
            stage_info.append({"start": remainder_start,
                               "end": remainder_start + eff_batchsize,
                               "num_shards": ssize,
                               "shard_id": srank})

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
        rbytes, wbytes = stage_batch_data(stage_comm, executor, shardinfo, files_shard, filesize, blocksize, target_directory, use_direct_io)
        total_bytes_read += rbytes
        total_bytes_write += wbytes
    
    # global barrier
    instance_comm.Barrier()

    return total_bytes_read, total_bytes_write
    

def stage_data_helper(global_comm, num_instances, instance_id, instance_comm,
                      local_size, local_rank, pargs, verify=False, 
                      full_dataset_per_node=True, use_direct_io=False,
                      seed=333,
                      prepare_staging = False):
    # - Every instance needs all the data, so we need inum replicas.
    # - Every rank irank within an instance can stage data_size / isize of the total data
    # - Since there are num_instances ranks working on the same data, we could shard this among
    # those ranks too
    gsize = global_comm.Get_size()
    grank = global_comm.Get_rank()
    isize = instance_comm.Get_size()
    irank = instance_comm.Get_rank()
    lsize = local_size
    lrank = local_rank
    
    # create staging filter:
    if (pargs.data_format == "dali-numpy") or (pargs.data_format == 'dali-es'):
        stage_filter_list = ['validation/data-*.npy', 'validation/label-*.npy',
                             'train/data-*.npy', 'train/label-*.npy']
    elif pargs.data_format == "dali-dummy":
        return
    else:
        raise NotImplementedError(f"Error, data-format {pargs.data_format} not implemented for staging")

    # create subdirectory for each instance, just in case if multiple instances see the same directory
    stage_dir = os.path.join(pargs.stage_dir_prefix, f"instance{instance_id}")
    if lrank == 0:
        os.makedirs(stage_dir, exist_ok = True)
    
    # split the global communicator according to irank: key could be instance_id but we would end up
    # with having all rank 0 on the same node. Now we got:
    # irank = 0: [0, 1, 2, ..... num_instances]
    # irank = 1: [1, 2, 3, ..... 0]]
    #keys = np.roll(np.arange(0, num_instances), irank).tolist()
    #stage_comm = global_comm.Split(color=irank, key=keys[instance_id])
    stage_comm = global_comm.Split(color=irank, key=instance_id)
    
    # split the instance by nodes and create a comm with all matching local ranks by node
    num_nodes_per_instance = isize // lsize
    instance_node_id = irank // lsize
    instance_node_comm = instance_comm.Split(color=lrank, key=instance_node_id)
    
    # stage the statsfile, it is OK to do that beforehand:
    if prepare_staging:
        if grank == 0:
            print("Copying stats.h5", flush=True)
            with open(os.path.join(pargs.data_dir_prefix, "stats.h5"), "rb") as f:
                statsfile = f.read()
        else:
            statsfile = None

        # broadcast the statsfile
        statsfile = global_comm.bcast(statsfile, 0)

        # save it
        if lrank == 0:
            with open(os.path.join(stage_dir, "stats.h5"), "wb") as f:
                f.write(statsfile)

    
    # iterate over staging filters
    file_stats = {}
    for stage_filter in stage_filter_list:
        
        if not prepare_staging:
            nvtx.range_push(f"stage {stage_filter}")
            if grank == 0:
                print(f"Staging {stage_filter}", flush=True)
        else:
            if grank == 0:
                print(f"Preparing file lists for {stage_filter}", flush=True)

        # get directories
        stage_source_directory = os.path.join(pargs.data_dir_prefix, os.path.dirname(stage_filter))
        stage_target_directory = os.path.join(stage_dir, os.path.dirname(stage_filter))

        # create target directory if not exist:
        if local_rank == 0:
            os.makedirs(stage_target_directory, exist_ok = True)
        
        # get file info to everybody
        if grank == 0:
            allfiles = sorted(glob(os.path.join(stage_source_directory, os.path.basename(stage_filter))))
        else:
            allfiles = None

        # shuffle files if requested
        if (grank == 0) and (not full_dataset_per_node) and (seed is not None):
            rng = np.random.default_rng(seed)
            rng.shuffle(allfiles)
            
        # communicate list of files
        allfiles = global_comm.bcast(allfiles, 0)
        
        # now stage the data so that each rank in each instance has the relevant data
        stage_start = time.perf_counter()
        total_read, total_write = stage_instance_data(stage_comm, instance_comm, instance_node_comm,
                                                      lsize, lrank, 
                                                      allfiles, stage_target_directory,
                                                      pargs.stage_batch_size,
                                                      pargs.stage_num_workers,
                                                      pargs.stage_mode,
                                                      full_dataset_per_node,
                                                      use_direct_io,
                                                      prepare_staging)
        stage_stop = time.perf_counter()

        # updating file stats buffer
        file_stats[stage_filter] = len(allfiles) 

        # skip the rest if we want to prep staging only
        if prepare_staging:
            continue
        
        # unit conversion
        unit_convert_gb = 1./float(1024*1024*1024)

        # allreduce:
        total_read = global_comm.allreduce(total_read)
        total_write = global_comm.allreduce(total_write)

        # convert units
        total_read *= unit_convert_gb
        total_write *= unit_convert_gb
        
        # stage duration:
        stage_duration = stage_stop - stage_start
        
        # print
        if grank == 0:
            print(f"""Staging {stage_filter} done.
                      Total number of files: {file_stats[stage_filter]}.
                      Elapsed time {stage_duration:.2f}s. 
                      Read {total_read:.2f} GB (bandwidth: {total_read/stage_duration:.2f} GB/s).
                      Write {total_write:.2f} GB (bandwidth: {total_write/stage_duration:.2f} GB/s).
                   """)

        # verify staging results if requested
        if verify:
            nvtx.range_push(f"stage_verify")
            if local_rank == 0:
                files = glob(os.path.join(stage_target_directory, os.path.basename(stage_filter)))
            else:
                files = []

            if not full_dataset_per_node:
                # if every node hosts a shard, we need to sum the results, if not we need to make sure everybody has the same
                files_full = instance_comm.allgather(files)
                files_full = set(itertools.chain(*files_full))
            else:
                files_full = set(files)
            num_files = len(files_full)

            # strip off the directory
            checkfiles1 = sorted([os.path.basename(x) for x in files_full])
            checkfiles2 = sorted([os.path.basename(x) for x in allfiles])

            assert(num_files == file_stats[stage_filter])
            assert(checkfiles1 == checkfiles2)
                
            if irank == 0:
                print(f"Staged data for {stage_filter}: {num_files}, expected: {file_stats[stage_filter]}", flush=True)
            nvtx.range_pop()

        # close range
        nvtx.range_pop()
            

    # make sure we have the right number of files everywhere
    assert(file_stats['validation/data-*.npy'] == file_stats['validation/label-*.npy'])
    assert(file_stats['train/data-*.npy'] == file_stats['train/label-*.npy'])
    
    return file_stats['train/data-*.npy'], file_stats['validation/data-*.npy']
