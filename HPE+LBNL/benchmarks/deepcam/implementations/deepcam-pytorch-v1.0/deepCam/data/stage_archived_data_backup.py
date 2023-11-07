import os
from glob import glob
import io
import tarfile
import copy
import itertools
import numpy as np
import argparse as ap
import concurrent.futures as cf
import time
from queue import LifoQueue as Queue
import torch.cuda.nvtx as nvtx


def load_file(filename, nbytes=0, offset=0):
    with open(filename, "rb") as f:
        f.seek(offset)
        if nbytes > 0:
            fdata = f.read()
        else:
            fdata = f.read(nbytes)
    return filename, fdata, nbytes, offset


def load_file_parallel(executor, num_shards, filename):
    nvtx.range_push("load_file_parallel")
    # split the data across ranks
    with tarfile.open(filename):
        files = os.path.get_names(filename)
    numfiles = ()
        
    numfiles_per_shard = int((numfiles + num_shards - 1) // num_shards)

    # submit loads
    handles = []
    for tid in range(num_workers):
        start = nbytes_per_shard * tid
        end = min(start + nbytes_per_shard, nbytes)
        nbytes_eff = end - start
        handles.append(executor.submit(load_file, filename, nbytes_eff, start))

    return handles


def save_file(ofname, fdata, offset=0):
    with open(ofname, "wb") as f:
        f.seek(offset)
        f.write(fdata)
    return len(fdata)


def save_file_parallel(handles, ofname):
    total_write_bytes = 0
    with open(ofname, 'wb') as f:  
        for future in cf.as_completed(handles):
            _, fdata, nbytes, offset = future.result()
            f.seek(offset)
            f.write(fdata)
            total_write_bytes += nbytes

    # get the members:
    with tarfile.open(ofname) as tar:
        files_shard = tar.get_names()
        
    return files_shard, total_write_bytes


# this routine stages data for each instance
def stage_instance_data(instance_comm, instance_node_comm,
                        lsize, lrank,
                        files, target_directory,
                        batch_size=1,
                        stage_num_workers=1,
                        stage_mode="node",
                        full_dataset_per_node=True,
                        prepare_staging=False):

    # comm parameters
    isize = instance_comm.Get_size()
    irank = instance_comm.Get_rank()
    nsize = instance_node_comm.Get_size()
    nrank = instance_node_comm.Get_rank()

    # each rank checks which files are inside
    shard_archive = [x for x in files if (f"rank{irank}.tar" in x)][0]
    if prepare_staging:

        # check the files in the tarball:
        with tarfile.open(shard_archive, 'r') as tar:
            files_shard = tar.getnames()

        # create a local communicator
        local_comm = instance_comm.Split(color=(irank // lsize), key=lrank)
        files_print = local_comm.allgather(files_shard)
        files_print = list(itertools.chain(*files_print))

        # split by label and data files:
        for tag in ["data", "label"]:
            fname = os.path.join(target_directory, f"files_{tag}.lst")
            files_tag = sorted([x for x in files_print if x.startswith(tag)])
        
            # create tags
            if lrank == 0:
                with open(fname, "w") as f:
                    f.write("\n".join(files_tag))
                    
        return 0, 0, files_shard

    # compute shard sizes
    executor = cf.ThreadPoolExecutor(max_workers=stage_num_workers)
    
    # read archive
    handles = load_file_parallel(executor, batch_size, shard_archive)
 
    # store it locally
    ofname = os.path.join(target_directory, os.path.basename(shard_archive))
    files_shard, total_bytes_write = save_file_parallel(handles, ofname)
    total_bytes_read = total_bytes_write
    
    # global barrier
    instance_comm.Barrier()

    return total_bytes_read, total_bytes_write, files_shard
    

def stage_archived_data_helper(global_comm, num_instances, instance_id, instance_comm,
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
    if (pargs.data_format == 'dali-numpy') or (pargs.data_format == 'dali-es'):
        stage_filter_list = [f'validation_archive_nranks{isize}/archive_rank*.tar', 
                             f'train_archive_nranks{isize}/archive_rank*.tar']
        stage_filter_list_extracted = ['validation/data-*.npy', 'validation/label-*.npy',
                                       'train/data-*.npy', 'train/label-*.npy']
    elif pargs.data_format == "dali-dummy":
        return
    else:
        raise NotImplementedError(f"Error, data-format {pargs.data_format} not implemented for staging")

    # only instance staging is supported in that mode
    assert(pargs.stage_mode == "instance")

    # create subdirectory for each instance, just in case if multiple instances see the same directory
    stage_dir = os.path.join(pargs.stage_dir_prefix, f"instance{instance_id}")
    if lrank == 0:
        os.makedirs(stage_dir, exist_ok = True)
    
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
        stage_source_directory = os.path.join(pargs.data_dir_prefix, os.path.dirname(stage_filter))
        stage_target_directory = os.path.join(stage_dir, os.path.dirname(stage_filter).split("_")[0])

        # create target directory if not exist:
        if local_rank == 0:
            os.makedirs(stage_target_directory, exist_ok = True)
        
        # get file info to everybody
        if grank == 0:
            allarchives = sorted(glob(os.path.join(stage_source_directory, os.path.basename(stage_filter))))
        else:
            allarchives = None
            
        # communicate list of files
        allarchives = global_comm.bcast(allarchives, 0)

        # check if the number of files is equal to the instance size:
        assert(len(allarchives) == isize)
        
        # now stage the data so that each rank in each instance has the relevant data
        stage_start = time.perf_counter()
        total_read, total_write, allfiles = stage_instance_data(instance_comm, instance_node_comm,
                                                                lsize, lrank, 
                                                                allarchives, stage_target_directory,
                                                                pargs.stage_batch_size,
                                                                pargs.stage_num_workers,
                                                                pargs.stage_mode,
                                                                full_dataset_per_node,
                                                                prepare_staging)
        stage_stop = time.perf_counter()

        # updating file stats buffer
        allfiles = instance_comm.allgather(allfiles)
        allfiles = list(itertools.chain(*allfiles))
        phase = stage_filter.split("_")[0]
        file_stats[f"{phase}/data-*.npy"] = len([x for x in allfiles if x.startswith("data-")])
        file_stats[f"{phase}/label-*.npy"] = len([x for x in allfiles if x.startswith("label-")])

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
                      Total number of data files: {file_stats[f"{phase}/data-*.npy"]}.
                      Total number of label files: {file_stats[f"{phase}/label-*.npy"]}.
                      Elapsed time {stage_duration:.2f}s. 
                      Read {total_read:.2f} GB (bandwidth: {total_read/stage_duration:.2f} GB/s).
                      Write {total_write:.2f} GB (bandwidth: {total_write/stage_duration:.2f} GB/s).
                   """)

        # pop range
        nvtx.range_pop() 

            
    # verify staging results if requested
    if verify and not prepare_staging:
        nvtx.range_push(f"stage_verify")
        for stage_filter in stage_filter_list_extracted:
            if local_rank == 0:
                files_staged = glob(os.path.join(stage_target_directory, os.path.basename(stage_filter)))
                tag = os.path.basename(stage_filter).split("-")[0]
                fname = os.path.join(stage_dir, os.path.dirname(stage_filter), f"files_{tag}.lst")
                with open(fname, 'r') as f:
                    files_list = f.read().split()
            else:
                files_staged = []
                files_list = []
                
        if not full_dataset_per_node:
            # if every node hosts a shard, we need to sum the results, if not we need to make sure everybody has the same
            files_full_staged = instance_comm.allgather(files_staged)
            files_full_staged = set(itertools.chain(*files_full_staged))
            files_full_list = instance_comm.allgather(files_list)
            files_full_list = set(itertools.chain(*files_full_list))
            
        else:
            files_full_staged = set(files_staged)
            files_full_list = set(files_List)
            num_files = len(files_full_staged)
            
        # strip off the directory
        checkfiles1 = sorted([os.path.basename(x) for x in files_full_staged])
        checkfiles2 = sorted([os.path.basename(x) for x in files_full_list])
        
        assert(num_files == file_stats[stage_filter])
        assert(checkfiles1 == checkfiles2)
                
        if irank == 0:
            print(f"Staged data for {stage_filter}: {num_files}, expected: {file_stats[stage_filter]}", flush=True)
        nvtx.range_pop()

    # sync up here
    global_comm.Barrier()

    # make sure we have the right number of files everywhere
    assert(file_stats['validation/data-*.npy'] == file_stats['validation/label-*.npy'])
    assert(file_stats['train/data-*.npy'] == file_stats['train/label-*.npy'])
    
    return file_stats['train/data-*.npy'], file_stats['validation/data-*.npy']
