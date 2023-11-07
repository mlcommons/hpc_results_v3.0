import os
import glob
import argparse
import math
import numpy as np
import tensorflow as tf

from mpi4py import MPI


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', help="Directory which contains the input data", required=True)
    parser.add_argument('-o', '--output-dir', help="Directory which will hold the output data", required=True)
    parser.add_argument('-p', '--num-processes', default=4, help="Number of processes to spawn for file conversion")
    parser.add_argument('-c', '--compression', default=None, help="Compression Type.")
    return parser.parse_args()


def filter_func(item, lst):
    item = os.path.basename(item).replace(".tfrecord", ".npy")
    return item not in lst


def convert_numpy(record, ofname_data, ofname_label, preprocess = False, transpose = False):
    
    feature_spec = dict(x=tf.io.FixedLenFeature([], tf.string),
                        y=tf.io.FixedLenFeature([4], tf.float32))
        
    example = tf.io.parse_single_example(record, features=feature_spec)
    
    # data
    data = tf.io.decode_raw(example['x'], tf.int16)
    data = tf.reshape(data, (128, 128, 128, 4))
    
    if transpose:
        data = tf.transpose(data, perm=[3, 0, 1, 2])
        
    if preprocess:
        data = tf.math.log(data + 1.)

    # label
    label = example['y']

    # get numpy tensors
    data_arr = data.numpy()
    label_arr = label.numpy()

    # save
    np.save(ofname_data, data_arr)
    np.save(ofname_label, label_arr)


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
            os.makedirs(args.output_dir, exist_ok = True)
    comm.barrier()
    
    # get input files and split
    inputfiles_all = glob.glob(os.path.join(args.input_dir, '*.tfrecord'))
    total = len(inputfiles_all)
    chunksize = math.ceil(total // comm_size)
    start = comm_rank * chunksize
    end = min([total, start + chunksize])
    inputfiles_all = inputfiles_all[start:end]

    # remove the files which are done
    filesdone_label = set([os.path.basename(x).replace("_label", "") for x in glob.glob(os.path.join(args.output_dir, '*_label.npy'))])
    filesdone_data = set([os.path.basename(x).replace("_data", "") for x in glob.glob(os.path.join(args.output_dir, '*_data.npy'))])
    filesdone = set([x for x in filesdone_label if x in filesdone_data])

    # compute local inputfiles
    inputfiles = list(filter(lambda x: filter_func(x, filesdone), inputfiles_all))

    # report the findings
    print(f"Found {len(inputfiles_all)} files, {len(filesdone)} are done, {len(inputfiles)} are remaining.")

    # assign gpu
    gpulist = tf.config.experimental.list_physical_devices('GPU')
    num_gpu = len(gpulist)
    comm_local_rank = comm_rank % num_gpu
    tf.config.experimental.set_visible_devices(gpulist[comm_local_rank], 'GPU')
    
    # set tfrecord dataset
    dataset = tf.data.TFRecordDataset(inputfiles, compression_type = args.compression)

    for ifname, record in zip(inputfiles, dataset.as_numpy_iterator()):
        ofname_data = os.path.join(args.output_dir, os.path.basename(ifname).replace(".tfrecord", "_data.npy"))
        ofname_label = os.path.join(args.output_dir, os.path.basename(ifname).replace(".tfrecord", "_label.npy"))
        convert_numpy(record, ofname_data, ofname_label)
    
    # wait for others
    comm.barrier()
    
    
if __name__ == "__main__":
    main()
