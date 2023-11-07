# 'Regression of 3D Sky Map to Cosmological Parameters (CosmoFlow)'
# Copyright (c) 2018, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
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
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.
#
# NOTICE.  This Software was developed under funding from the U.S. Department of
# Energy and the U.S. Government consequently retains certain rights. As such,
# the U.S. Government has been granted for itself and others acting on its
# behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
# to reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit other to do so.

"""
Data preparation script which reads GZIP compressed TFRecord files and produces decompressed TFRecord files.
"""

# System
import os
import argparse
import logging
from functools import partial

# Externals
import numpy as np
import tensorflow as tf

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir',
        default='/project/projectdirs/m3363/www/cosmoUniverse_2019_05_4parE')
    parser.add_argument('-o', '--output-dir',
        default='/global/cscratch1/sd/sfarrell/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--max-files', type=int)
    parser.add_argument('--task', type=int, default=os.environ.get('OMPI_COMM_WORLD_RANK', 0))
    parser.add_argument('--n-tasks', type=int, default=os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    return parser.parse_args()

def find_files(input_dir, max_files=None):
    files = []
    for subdir, _, subfiles in os.walk(input_dir):
        for f in subfiles:
            if f.endswith('tfrecord'):
                files.append(os.path.join(subdir, f))
    files = sorted(files)
    if max_files is not None:
        files = files[:max_files]
    return files

def write_record(output_file, example, compression=None):
    with tf.io.TFRecordWriter(output_file, options=compression) as writer:
        writer.write(example.SerializeToString())

def _parse_data(sample_proto):
    # Parse the serialized features
    feature_spec = dict(x=tf.io.FixedLenFeature([], tf.string),
                        y=tf.io.FixedLenFeature([4], tf.float32))
    parsed_example = tf.io.parse_single_example(
        sample_proto, features=feature_spec)

    # Decode the bytes data
    x = tf.io.decode_raw(parsed_example['x'], tf.int16)
    y = parsed_example['y']
    return x, y

def process_file(input_files, output_dir):
    dataset = tf.data.TFRecordDataset(input_files, compression_type='GZIP')
    dataset = dataset.map(_parse_data)

    # Loop over sub-volumes
    for input_file, (x, y) in zip(input_files, dataset):
        #logging.info('Reading %s', input_file)

        # Output file name pattern. To avoid name collisions,
        # we prepend the subdirectory name to the output file name.
        # We also append the subvolume index
        subdir = os.path.basename(os.path.dirname(input_file))
        output_file = os.path.join(
            output_dir,
            subdir,
            os.path.basename(input_file)
        )

        # Convert to TF example
        x = x.numpy()
        feature_dict = dict(
            x=tf.train.Feature(bytes_list=tf.train.BytesList(value=[x.tobytes()])),
            y=tf.train.Feature(float_list=tf.train.FloatList(value=y)))
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        # Write the output file
        logging.info('Writing %s', output_file)
        write_record(output_file, tf_example, compression=None)

def main():
    """Main function"""

    # Parse the command line
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info('Initializing')

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)

    # Select my subset of input files
    input_files = find_files(args.input_dir, max_files=args.max_files)
    input_files = np.array_split(input_files, args.n_tasks)[args.task]

    # Process input files with a worker pool
    process_file(input_files, args.output_dir)

    logging.info('All done!')

if __name__ == '__main__':
    main()
