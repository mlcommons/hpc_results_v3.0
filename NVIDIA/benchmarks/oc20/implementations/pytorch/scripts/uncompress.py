# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

"""
Uncompresses downloaded S2EF datasets to be used by the LMDB preprocessing
script - preprocess_ef.py
"""

import argparse
import glob
import lzma
import multiprocessing as mp
import os

from tqdm import tqdm


def read_lzma(inpfile, outfile):
    with open(inpfile, "rb") as f:
        contents = lzma.decompress(f.read())
        with open(outfile, "wb") as op:
            op.write(contents)


def decompress_list_of_files(ip_op_pair):
    ip_file, op_file = ip_op_pair
    read_lzma(ip_file, op_file)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipdir", type=str, help="Path to compressed dataset directory")
    parser.add_argument("--opdir", type=str, help="Directory path to uncompress files to")
    parser.add_argument("--num-workers", type=int, help="# of processes to parallelize across")
    return parser


def main(args):
    os.makedirs(args.opdir, exist_ok=True)

    filelist = glob.glob(os.path.join(args.ipdir, "*txt.xz")) + glob.glob(os.path.join(args.ipdir, "*extxyz.xz"))
    ip_op_pairs = []
    for i in filelist:
        fname_base = os.path.basename(i)
        ip_op_pairs.append((i, os.path.join(args.opdir, fname_base[:-3])))

    pool = mp.Pool(args.num_workers)
    list(
        tqdm(
            pool.imap(decompress_list_of_files, ip_op_pairs),
            total=len(ip_op_pairs),
            desc=f"Uncompressing {args.ipdir}",
        )
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
