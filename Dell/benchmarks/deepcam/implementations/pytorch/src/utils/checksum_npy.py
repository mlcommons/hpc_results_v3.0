# The MIT License (MIT)
#
# Modifications Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
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
import glob
import numpy as np
import math
from itertools import chain
import argparse as ap
import hashlib
from mpi4py import MPI


def compute_checksum(ifname):
    with open(ifname, 'rb') as f:
        token = f.read()

    return hashlib.sha256(token).hexdigest()
        

def main(args):

    # get rank
    comm = MPI.COMM_WORLD.Dup()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # get input files
    inputfiles_all = glob.glob(os.path.join(args.input_directory, "*[.npy|.h5]"))

    # exit if we did not find any files
    if not inputfiles_all:
        if comm_rank == 0:
            print("No files found, exiting!")
        comm.barrier()
        return

    # wait for everybody to be done
    comm.barrier()
    
    if comm_rank == 0 :
        print(f"Checking {len(inputfiles_all)} files.")
            
    # split across ranks: round robin
    inputfiles_local = []
    for idx, ifname in enumerate(inputfiles_all):
        if idx % comm_size == comm_rank:
            inputfiles_local.append(ifname)
    
    # checksum files
    checksums = []
    for ifname in inputfiles_local:
        checksums.append((os.path.basename(ifname), compute_checksum(ifname)))
        
    # gather
    checksums = comm.gather(checksums, root=0)

    # flatten list:
    if comm_rank == 0:
        print(f"Writing Checksum File {args.checksum_file}")
        
        checksums = list(chain(*checksums))
        checksums = sorted(checksums, key=lambda x: x[0])
        
        # convert to strings
        checksums = [x[0] + " " + x[1] for x in checksums]

        # write out:
        with open(args.checksum_file, "w") as f:
            f.write("\n".join(checksums))

        print("done")

    # wait for the others
    comm.barrier()


if __name__ == "__main__":
    
    AP = ap.ArgumentParser()
    AP.add_argument("--input_directory", type=str, help="Directory with input files", required = True)
    AP.add_argument("--checksum_file", type=str, help="Directory with checksum file", required = True)
    pargs = AP.parse_args()
    
    main(pargs)
