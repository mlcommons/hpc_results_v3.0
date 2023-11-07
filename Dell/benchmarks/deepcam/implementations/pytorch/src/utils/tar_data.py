# The MIT License (MIT)
#
# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
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
import sys
import numpy as np
import tarfile
import argparse as ap
import shutil
import torch
from mpi4py import MPI


def main(args):

    overwrite = args.overwrite
    shuffle = not args.disable_shuffle
    data_format = "nhwc"
    input_path = args.input_directory
    output_path = args.output_directory
    full_dataset_per_node = args.full_dataset_per_node
    
    #MPI
    comm = MPI.COMM_WORLD.Dup()
    comm_rank = comm.rank
    comm_size = comm.size
    if torch.cuda.is_available():
        comm_local_size = torch.cuda.device_count()
    else:
        comm_local_size = 1
    comm_local_rank = comm_rank % comm_local_size
    cycle_length = comm_local_size

    # copy the statsfile first:
    if comm_rank == 0:
        shutil.copy(os.path.join(input_path, "stats.h5"), os.path.join(output_path, "stats.h5"))    

    #now tar the data files:
    for phase in ["validation", "train"]:

        # data directories
        input_root = os.path.join(input_path, phase)
        output_root = os.path.join(output_path, phase + f"_archive_nranks{comm_size}")

        # create output dir
        if (comm_rank == 0) and (not os.path.isdir(output_root)):
            os.makedirs(output_root, exist_ok=True)
        comm.Barrier()

        # check if outputfile exists
        tarfilename = os.path.join(output_root, f"archive_rank{comm_rank}.tar")
        if os.path.isfile(tarfilename) and tarfile.is_tarfile(tarfilename) and not overwrite:
            print(f"File {tarfilename} already exists, skipping")
            continue    
        
        #get files
        allfiles = [ x.replace('data-', '') for x in os.listdir(input_root) \
                     if x.endswith('.npy') and x.startswith('data-') ]

        # shuffle if requested
        if shuffle:
            np.random.shuffle(allfiles)

        #split list
        numfiles = len(allfiles)
        if not full_dataset_per_node:
            chunksize = numfiles // comm_size
            start = chunksize * comm_rank
            end = min([start + chunksize, numfiles])
            files_remainder = allfiles[comm_size * chunksize:]
            files = allfiles[start:end].copy()
    
            # deadl with remainder:
            cycle_offset = 0
            for idf, fname in enumerate(files_remainder):
                if ((idf + cycle_offset) % comm_size == comm_rank):
                    files.append(fname)
                cycle_offset += cycle_length
        else:
            chunksize = numfiles // comm_local_size
            start = chunksize * comm_local_rank
            end = min([start + chunksize, numfiles])
            files_remainder = allfiles[comm_local_size * chunksize:]
            files = allfiles[start:end].copy()

            for idf, fname in enumerate(files_remainder):
                if (idf == comm_local_rank):
                    files.append(fname)

        #now each rank is tarring their files, data and label:
        files_tar = [os.path.join(input_root, "data-" + x) for x in files] + [os.path.join(input_root, "label-" + x) for x in files]
    
        # create tarfile
        if comm_rank == 0:
            print(f"Starting archiving {phase}")
        with tarfile.open(tarfilename, "w") as tar:
            for fname in files_tar:
                arcname=os.path.basename(fname)
                tar.add(fname, arcname=arcname)
        comm.Barrier()
        if comm_rank == 0:
            print(f"Finished archiving {phase}")
    


if __name__ == "__main__":
    AP = ap.ArgumentParser()
    AP.add_argument("--input_directory", type=str, help="Directory with input files", required = True)
    AP.add_argument("--output_directory", type=str, help="Directory with output files.", required = True)
    AP.add_argument("--disable_shuffle", action="store_true", help="Do not initially shuffle the data")
    AP.add_argument("--overwrite", action="store_true", help="Overwrite data if already existing")
    AP.add_argument("--full_dataset_per_node", action="store_true", help="Stage a full copy of the dataset per node")
    args = AP.parse_args()
    
    main(args)
