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

import functools
import os
import glob
import argparse
import math
import numpy as np
import tensorflow as tf

from mpi4py import MPI
import tempfile
import gzip
import multiprocessing as mp


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input-dir', help="Directory which contains the input data", required=True)
    parser.add_argument(
        '-o', '--output-dir', help="Directory which will hold the output data", required=True)
    parser.add_argument('-p', '--num-processes', default=4,
                        help="Number of processes to spawn for file conversion")
    parser.add_argument('-c', '--compression', default=None,
                        help="Compression Type.")
    return parser.parse_args()


def filter_func(item, lst):
    item = os.path.basename(item).replace(".tfrecord", ".tfrecord.gzip")
    return item not in lst


tmp_file = tempfile.mkstemp()


def convert(record, ofname_data):

    feature_spec = dict(x=tf.io.FixedLenFeature([], tf.string),
                        y=tf.io.FixedLenFeature([4], tf.float32))

    example = tf.io.parse_single_example(record, features=feature_spec)
    data = tf.io.decode_raw(example['x'], tf.int16)
    data = tf.reshape(data, (128, 128, 128, 4))

    example = tf.train.Example(features=tf.train.Features(
        feature={
            "x": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.numpy().tobytes()])),
            "y": tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(example['y'].numpy())))
        }
    ))

    with tf.io.TFRecordWriter(tmp_file[1]) as file_writer:
        file_writer.write(example.SerializeToString())

    with gzip.open(ofname_data, "wb", compresslevel=7) as fdst, open(tmp_file[1], "rb") as fsrc:
        fdst.write(fsrc.read())


def main():
    """Main function"""
    # Parse the command line
    args = parse_args()

    # get rank
    comm = MPI.COMM_WORLD.Dup()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # create output dir
    if comm_rank == 0:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
    comm.barrier()

    # get input files and split
    inputfiles_all = glob.glob(os.path.join(args.input_dir, '*.tfrecord'))
    total = len(inputfiles_all)
    chunksize = math.ceil(total // comm_size)
    start = comm_rank * chunksize
    end = min([total, start + chunksize])
    inputfiles_all = inputfiles_all[start:end]

    tf.config.experimental.set_visible_devices([], 'GPU')

    # set tfrecord dataset
    dataset = tf.data.TFRecordDataset(
        inputfiles_all, compression_type=args.compression)

    i = 0
    for ifname, record in zip(inputfiles_all, dataset.as_numpy_iterator()):
        ofname_data = os.path.join(args.output_dir, os.path.basename(
            ifname))
        convert(record, ofname_data)

        i += 1
        if i % 20 == 0:
            print(f"[{comm_rank}] Copied {i}/{len(inputfiles_all)}", flush=True)

    # wait for others
    comm.barrier()


if __name__ == "__main__":
    main()
