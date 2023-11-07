import os
from glob import glob
import itertools
import numpy as np
import argparse as ap
import concurrent.futures as cf
import time
from queue import LifoQueue as Queue
import torch.cuda.nvtx as nvtx
import copy
import tarfile


def finalize_load_local(fdata_handles):
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


def finalize_load_global(comm, files, rbytes):
    nvtx.range_push("finalize_load_global")
    files_all = exchange_buffers(comm, files)
    rbytes = comm.allreduce(rbytes)
    nvtx.range_pop()
    
    return files_all, rbytes


def load_file(tar, filename):
    f = tar.extractfile(filename)
    fdata = f.read()
    
    return filename, fdata, len(fdata)


def load_batch_parallel(executor, tar, files, comm_size, comm_rank):
    nvtx.range_push("load_batch_parallel")
    # split the data across ranks
    if comm_size > 1:
        num_files = len(files)
        num_files_per_shard = (num_files + comm_size - 1) // comm_size
        start = num_files_per_shard * comm_rank
        end = min(start + num_files_per_shard, num_files)
        files_load = files[start:end]
    else:
        files_load = files

    # submit loads:
    queue = Queue()
    for filename in files_load:
        queue.put(executor.submit(load_file, tar, filename))
    nvtx.range_pop()
    
    return queue


def finalize_save_local(fdata_handles):
    nvtx.range_push("finalize_save_local")
    wbytes = 0
    while not fdata_handles.empty():
        handle = fdata_handles.get()
        if handle.done():
            fbytes = handle.result()
            wbytes += fbytes
        else:
            fdata_handles.put(handle)
    nvtx.range_pop()

    return wbytes 
        

def save_file(ofname, fdata):
    with open(ofname, "wb") as f:
        f.write(fdata)
    return len(fdata)


def save_batch_parallel(executor, output_dir, fdata):
    nvtx.range_push("save_batch_parallel")
    queue = Queue()
    for fn, fd in fdata:
        ofn = os.path.join(output_dir, os.path.basename(fn))
        queue.put(executor.submit(save_file, ofn, copy.copy(fd)))
    nvtx.range_pop()

    return queue


def save_batch(output_dir, fdata):
    nvtx.range_push("save_batch")
    wbytes = 0.
    for fn, fd in fdata:
        ofn = os.path.join(output_dir, os.path.basename(fn))
        wbytes += save_file(ofn, fd)
    nvtx.range_pop()
    
    return wbytes


def exchange_buffers(comm, fdata):
    nvtx.range_push("exchange_buffers")
    # quick exit if we do not need to communicate
    if comm.Get_size() == 1:
        return fdata

    # determine how many files everyone got:
    num_files = len(fdata)
    num_files = comm.allgather(num_files)

    #print(f"{comm.Get_rank()} num_files: {num_files}")
    
    # gather data
    if True:
        # pickle method
        # gather
        fdata_all = comm.allgather(fdata)
        # flatten
        fdata_result = list(itertools.chain(*fdata_all))
    else:
        # pack method
        # pack 
        fname_gather = np.stack([np.frombuffer(x.encode('utf-8'), dtype='S1', count=len(x.encode('utf-8')), offset=0) for x,_ in fdata], axis=0)
        fdata_gather = np.stack([np.frombuffer(y, dtype='S1', count=len(y), offset=0) for _,y in fdata], axis=0)
        # gather
        fname_all = comm.allgather(fname_gather)
        fdata_all = comm.allgather(fdata_gather)
        # unpack/flatten
        fname_all = [y.tobytes().decode('utf-8') for y in list(itertools.chain(*[list(x) for x in fname_all]))]
        fdata_all = [y.tobytes() for y in list(itertools.chain(*[list(x) for x in fdata_all]))]
        # zip
        fdata_result = list(zip(fname_all, fdata_all))
    nvtx.range_pop()

    # return
    return fdata_result


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


# this routine stages data for each instance
def stage_instance_data(stage_comm, instance_comm, instance_node_comm,
                        lsize, lrank,
                        files,
                        source_filename,
                        target_directory,
                        batch_size=-1,
                        stage_num_workers=1,
                        stage_mode="node",
                        full_dataset_per_node=True,
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
        
        
    # now, let's take care of the data: update the number of files because of remainder
    num_files_per_shard = len(files_shard)
    if batch_size > 0:
        num_batches = (num_files_per_shard + batch_size - 1) // batch_size
    else:
        num_batches = 1
        batch_size = num_files_per_shard
        
    # create executor
    executor = cf.ThreadPoolExecutor(max_workers = stage_num_workers)
    tar = tarfile.open(source_filename, 'r:')
    
    # submit first batch
    fdata_handles = load_batch_parallel(executor, tar, files_shard[0:min(num_files_per_shard, batch_size)],
                                        ssize if stage_mode == "global" else 1,
                                        srank if stage_mode == "global" else 0)

    # batch loop: go one step further to process the last batch
    save_handles = []
    total_bytes_read = 0
    total_bytes_write = 0
    for bid in range(1, num_batches+1):
        # wait for data loader
        #print(f"{bid}: wait for data")
        #fdata_save, rbytes = fdata_handle.result()
        fdata_save, rbytes = finalize_load_local(fdata_handles)
        
        # finalize the load if required:
        if stage_mode == "global":
            fdata_save, rbytes = finalize_load_global(stage_comm, fdata_save, rbytes)
            
        # increment total bytes read
        total_bytes_read += rbytes
        
        # get next batch, only submit if there are any
        if bid < num_batches:
            batch_start = bid * batch_size
            batch_end = min(batch_start + batch_size, num_files_per_shard)
            fdata_handles = load_batch_parallel(executor, tar, files_shard[batch_start:batch_end],
                                                ssize if stage_mode == "global" else 1,
                                                srank if stage_mode == "global" else 0)

        # exchange buffers inside instance if necessary
        if full_dataset_per_node and (stage_mode != "node"):
            fdata_save = exchange_buffers(instance_node_comm, fdata_save)

        # wait till data is saved
        if save_handles:
            total_bytes_write += finalize_save_local(save_handles)
        
        # store locally
        #saved = executor.submit(save_batch, target_directory, fdata_save)
        save_handles = save_batch_parallel(executor, target_directory, fdata_save)

    # close the tarfile, not needed anymore
    tar.close()
        
    # finalize last batch
    if save_handles:
        total_bytes_write += finalize_save_local(save_handles)

    # global barrier
    instance_comm.Barrier()

    return total_bytes_read, total_bytes_write
    

def stage_data_helper(global_comm, num_instances, instance_id, instance_comm,
                      local_size, local_rank, pargs, verify=False, full_dataset_per_node=True, seed=333,
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

        nvtx.range_push(f"stage {stage_filter}")
        
        if not prepare_staging and (grank == 0):
            print(f"Staging {stage_filter}", flush=True)
        elif grank == 0:
            print(f"Preparing file lists for {stage_filter}", flush=True)

        # get directories
        stage_source_filename = os.path.join(pargs.data_dir_prefix, os.path.dirname(stage_filter)+ '.tar')
        stage_target_directory = os.path.join(stage_dir, os.path.dirname(stage_filter))

        # create target directory if not exist:
        if local_rank == 0:
            os.makedirs(stage_target_directory, exist_ok = True)
        
        # get file info to everybody
        if grank == 0:
            tag = os.path.basename(stage_filter).split("-")[0]
            with tarfile.open(stage_source_filename, 'r') as tar:
                allfiles = sorted([x for x in tar.get_names() if x.startswith(tag)])
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
                                                      allfiles,
                                                      stage_source_filename,
                                                      stage_target_directory,
                                                      pargs.stage_batch_size,
                                                      pargs.stage_num_workers,
                                                      pargs.stage_mode,
                                                      full_dataset_per_node,
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
